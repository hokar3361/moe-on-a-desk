// membench: DRAM read bandwidth on the GPU vs per-lane load width and per-lane bytes per stage.
// Each "stage" reads S bytes (a different region each time, total working set >> SLC); we time NST stages in one
// command buffer with a barrier between them (like the real graph), and report GB/s.
// build: clang -fobjc-arc -framework Metal -framework Foundation membench.m -o membench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
// each thread reads U units of type T at stride (grid size) and xors them
#define KER(name, T, ACC) \
kernel void name(device const T * p [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & n [[buffer(2)]], constant uint & U [[buffer(3)]], \
                 uint tid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) { \
    ACC a = 0; \
    for (uint i = 0; i < U; ++i) { const uint k = tid + i*nt; if (k < n) a ^= ACC(p[k]); } \
    if (all(a == ACC(0x12345678u))) out[tid] = 1; \
}
KER(r_u16, ushort, uint)
KER(r_u32, uint, uint)
KER(r_u64, uint2, uint2)
KER(r_u128, uint4, uint4)
// contiguous per thread: thread reads U consecutive units (like a lane walking its own chunk)
#define KERC(name, T, ACC) \
kernel void name(device const T * p [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & n [[buffer(2)]], constant uint & U [[buffer(3)]], \
                 uint tid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) { \
    ACC a = 0; \
    for (uint i = 0; i < U; ++i) { const uint k = tid*U + i; if (k < n) a ^= ACC(p[k]); } \
    if (all(a == ACC(0x12345678u))) out[tid] = 1; \
}
KERC(c_u16, ushort, uint)
KERC(c_u32, uint, uint)
KERC(c_u128, uint4, uint4)

// compile-time U (fully unrolled): thread reads U consecutive uints
#define KERU(name, UU) \
kernel void name(device const uint * p [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & n [[buffer(2)]], constant uint & U [[buffer(3)]], \
                 uint tid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) { \
    uint v[UU]; \
    _Pragma("unroll") for (uint i = 0; i < UU; ++i) { const uint k = tid*UU + i; v[i] = k < n ? p[k] : 0u; } \
    uint a = 0; _Pragma("unroll") for (uint i = 0; i < UU; ++i) a ^= v[i]; \
    if (a == 0x12345678u) out[tid] = 1; \
}
KERU(u_c4, 4)
KERU(u_c8, 8)
KERU(u_c16, 16)
KERU(u_c32, 32)
// strided within a simdgroup row: lane reads uints at lane + 32*i (coalesced across lanes), UU per lane
#define KERS(name, UU) \
kernel void name(device const uint * p [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & n [[buffer(2)]], constant uint & U [[buffer(3)]], \
                 uint tid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) { \
    const uint sg = tid/32, ln = tid%32; uint v[UU]; \
    _Pragma("unroll") for (uint i = 0; i < UU; ++i) { const uint k = sg*32*UU + ln + 32*i; v[i] = k < n ? p[k] : 0u; } \
    uint a = 0; _Pragma("unroll") for (uint i = 0; i < UU; ++i) a ^= v[i]; \
    if (a == 0x12345678u) out[tid] = 1; \
}
KERS(u_s4, 4)
KERS(u_s8, 8)
KERS(u_s16, 16)
KERS(u_s32, 32)
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const size_t S = argc > 1 ? (size_t) atof(argv[1])*1e6 : 4600000;   // bytes per stage
        const int NST = 64;
        const size_t total = S*NST;
        id<MTLBuffer> buf = [dev newBufferWithLength:total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:64<<20 options:MTLResourceStorageModeShared];
        struct { const char * name; int unit; int fixu; } ks[] = { {"c_u32",4,16},{"u_c4",4,4},{"u_c8",4,8},{"u_c16",4,16},{"u_c32",4,32},{"u_s4",4,4},{"u_s8",4,8},{"u_s16",4,16},{"u_s32",4,32} };
        const int upt_list[] = { 1 };
        for (int ki = 0; ki < 9; ++ki) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(ks[ki].name)] error:&err];
            for (int ui = 0; ui < 1; ++ui) {
                const uint U = ks[ki].fixu;
                const uint n = (uint) (S/ks[ki].unit);
                const uint nthr = (n + U - 1)/U;
                double best = 1e30;
                for (int rep = 0; rep < 5; ++rep) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:out offset:0 atIndex:1];
                    [enc setBytes:&n length:4 atIndex:2];
                    [enc setBytes:&U length:4 atIndex:3];
                    for (int st = 0; st < NST; ++st) {
                        [enc setBuffer:buf offset:(size_t) st*S atIndex:0];
                        [enc dispatchThreads:MTLSizeMake(nthr, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    const double t = cb.GPUEndTime - cb.GPUStartTime;
                    if (rep > 0 && t < best) best = t;
                }
                printf("%-7s U=%2u  %7.2f us/stage  %6.0f GB/s\n", ks[ki].name, U, best/NST*1e6, (double) S*NST/best/1e9);
            }
        }
    }
    return 0;
}
