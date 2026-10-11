// cosched: can a small dependent chain (draft-like) hide inside the stages of a big occupancy-limited chain
// (verify-like) when both are encoded in ONE concurrent encoder (each draft stage placed in a verify stage,
// the verify barriers ordering both chains)?
//   A: verify chain only           (NST stages, each a matvec-like kernel reading SB bytes, 800 TGs x 64)
//   B: draft chain only            (ND stages, each small: 64 TGs x 64, reading SD bytes)
//   C: A and B interleaved: draft stage i dispatched right after verify stage i*(NST/ND), before its barrier
// build: clang -fobjc-arc -framework Metal -framework Foundation cosched.m -o cosched
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
// each threadgroup reads its slice with a few dependent loads per thread (latency-bound like our matvecs)
kernel void k_read(device const uint * p [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & per_tg [[buffer(2)]],
                   uint tg [[threadgroup_position_in_grid]], ushort lid [[thread_index_in_threadgroup]], ushort nth [[threads_per_threadgroup]]) {
    device const uint * q = p + (uint64_t) tg*per_tg;
    uint a = 0;
    for (uint i = lid; i < per_tg; i += nth) { a = a*31u + q[i]; }
    if (a == 0x12345678u) out[tg] = a;
}
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_read"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const int NST = 2000, ND = 100;
        const uint VTG = 800, DTG = 64;
        const uint vper = 1024, dper = 2048;                 // uints per threadgroup: 3.3 MB per verify stage, 0.5 MB per draft stage
        const size_t vbytes = (size_t) VTG*vper*4, dbytes = (size_t) DTG*dper*4;
        const int NVB = 24;                                  // distinct verify buffers (working set >> SLC)
        id<MTLBuffer> vb = [dev newBufferWithLength:vbytes*NVB options:MTLResourceStorageModePrivate];
        id<MTLBuffer> db = [dev newBufferWithLength:dbytes*8 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:1<<20 options:MTLResourceStorageModeShared];
        for (int mode = 0; mode < 3; ++mode) {
            double best = 1e30;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:ps];
                [enc setBuffer:out offset:0 atIndex:1];
                const int nv = mode == 1 ? 0 : NST;
                for (int st = 0; st < (mode == 1 ? ND : nv); ++st) {
                    if (mode != 1) {
                        [enc setBuffer:vb offset:(size_t) (st % NVB)*vbytes atIndex:0];
                        [enc setBytes:&vper length:4 atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(VTG, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    }
                    if (mode == 1 || (mode == 2 && st % (NST/ND) == 0)) {
                        [enc setBuffer:db offset:(size_t) (st % 8)*dbytes atIndex:0];
                        [enc setBytes:&dper length:4 atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(DTG, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    }
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double t = cb.GPUEndTime - cb.GPUStartTime;
                if (rep > 0 && t < best) best = t;
            }
            const char * nm[] = { "A verify chain", "B draft chain", "C interleaved" };
            printf("%-16s %8.2f ms\n", nm[mode], best*1e3);
        }
    }
    return 0;
}
