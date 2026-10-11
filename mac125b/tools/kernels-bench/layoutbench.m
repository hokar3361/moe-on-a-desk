// layoutbench: does the 2-byte-aligned IQ block layout (and the 2-byte loads it forces) limit the matvec read speed?
// Real geometry of the MoE gate at B=1 (iq2_xxs): 10 experts x 640 rows, row = 10 blocks of 66 bytes (2 B scale +
// 64 B quants = 8 sub-blocks of 8 B). 800 threadgroups x 64 threads, 4 rows per simdgroup, lanes over the
// 80 sub-blocks of a row. No decode: loaded bytes are xor-folded (timing of the memory access pattern only).
//   A: AoS blocks as in GGUF, ushort loads (what the kernels do now)
//   B: same bytes re-laid out: per row 640 B of quants (16 B aligned) + 10 scales elsewhere; one uint2 per sub-block
//   C: like B, but a lane takes 2 adjacent sub-blocks with one uint4 load (16 lanes per row, 2 rows per step)
//   D: A plus the activation loads of the real kernel (32 floats per sub-block from a 10 KB vector)
//   E: B plus the same activation loads
// build: clang -fobjc-arc -framework Metal -framework Foundation layoutbench.m -o layoutbench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
constant uint NROW = 640, NEXP = 10, ROWB = 660, NSB = 80;

kernel void kA(device const uchar * w [[buffer(0)]], device uint * out [[buffer(1)]], device const float * y [[buffer(2)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const uint e = tg.x / 80, r0 = (tg.x % 80)*8 + sg*4;
    uint acc = 0;
    for (uint ib = ln; ib < NSB; ib += 32) {
        for (uint r = 0; r < 4; ++r) {
            device const uchar * row = w + ((uint64_t) e*NROW + r0 + r)*ROWB;
            device const ushort * blk = (device const ushort *) (row + 66*(ib/8));
            const ushort d = blk[0];
            device const ushort * q = blk + 1 + 4*(ib%8);
            acc ^= d ^ q[0] ^ (q[1] << 3) ^ (q[2] << 6) ^ (q[3] << 9);
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}
kernel void kB(device const uchar * w [[buffer(0)]], device uint * out [[buffer(1)]], device const float * y [[buffer(2)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const uint e = tg.x / 80, r0 = (tg.x % 80)*8 + sg*4;
    device const uchar * qbase = w;
    device const ushort * dbase = (device const ushort *) (w + (uint64_t) NEXP*NROW*640);
    uint acc = 0;
    for (uint ib = ln; ib < NSB; ib += 32) {
        for (uint r = 0; r < 4; ++r) {
            const uint64_t row = (uint64_t) e*NROW + r0 + r;
            const uint2 q = ((device const uint2 *) (qbase + row*640))[ib];
            const ushort d = dbase[row*10 + ib/8];
            acc ^= d ^ q.x ^ (q.y << 3);
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}
kernel void kC(device const uchar * w [[buffer(0)]], device uint * out [[buffer(1)]], device const float * y [[buffer(2)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const uint e = tg.x / 80, r0 = (tg.x % 80)*8 + sg*4;
    device const uchar * qbase = w;
    device const ushort * dbase = (device const ushort *) (w + (uint64_t) NEXP*NROW*640);
    uint acc = 0;
    const uint half_ = ln / 16, l16 = ln % 16;
    for (uint rr = 0; rr < 2; ++rr) {
        const uint r = 2*rr + half_;
        const uint64_t row = (uint64_t) e*NROW + r0 + r;
        for (uint i = l16; i < NSB/2; i += 16) {
            const uint4 q = ((device const uint4 *) (qbase + row*640))[i];
            const ushort d = dbase[row*10 + (2*i)/8];
            acc ^= d ^ q.x ^ (q.y << 3) ^ (q.z << 5) ^ (q.w << 7);
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}
kernel void kD(device const uchar * w [[buffer(0)]], device uint * out [[buffer(1)]], device const float * y [[buffer(2)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const uint e = tg.x / 80, r0 = (tg.x % 80)*8 + sg*4;
    uint acc = 0; float fa = 0;
    for (uint ib = ln; ib < NSB; ib += 32) {
        float yl[32];
        for (int i = 0; i < 32; ++i) yl[i] = y[32*ib + i];
        for (uint r = 0; r < 4; ++r) {
            device const uchar * row = w + ((uint64_t) e*NROW + r0 + r)*ROWB;
            device const ushort * blk = (device const ushort *) (row + 66*(ib/8));
            const ushort d = blk[0];
            device const ushort * q = blk + 1 + 4*(ib%8);
            const uint v = d ^ q[0] ^ (q[1] << 3) ^ (q[2] << 6) ^ (q[3] << 9);
            acc ^= v; fa += yl[v & 31];
        }
    }
    if (acc == 0x12345678u || fa == 1.2345f) out[0] = acc;
}
kernel void kE(device const uchar * w [[buffer(0)]], device uint * out [[buffer(1)]], device const float * y [[buffer(2)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
    const uint e = tg.x / 80, r0 = (tg.x % 80)*8 + sg*4;
    device const uchar * qbase = w;
    device const ushort * dbase = (device const ushort *) (w + (uint64_t) NEXP*NROW*640);
    uint acc = 0; float fa = 0;
    for (uint ib = ln; ib < NSB; ib += 32) {
        float yl[32];
        for (int i = 0; i < 32; ++i) yl[i] = y[32*ib + i];
        for (uint r = 0; r < 4; ++r) {
            const uint64_t row = (uint64_t) e*NROW + r0 + r;
            const uint2 q = ((device const uint2 *) (qbase + row*640))[ib];
            const ushort d = dbase[row*10 + ib/8];
            const uint v = d ^ q.x ^ (q.y << 3);
            acc ^= v; fa += yl[v & 31];
        }
    }
    if (acc == 0x12345678u || fa == 1.2345f) out[0] = acc;
}
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const size_t S = (size_t) 10*640*660 + 4096;     // one op's bytes (both layouts fit)
        const int NST = 64;                               // stages (different expert sets each)
        id<MTLBuffer> buf = [dev newBufferWithLength:S*NST options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLBuffer> yb  = [dev newBufferWithLength:2560*4 options:MTLResourceStorageModeShared];
        const char * names[] = { "kA", "kB", "kC", "kD", "kE" };
        for (int k = 0; k < 5; ++k) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(names[k])] error:&err];
            double best = 1e30;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:ps];
                [enc setBuffer:out offset:0 atIndex:1];
                [enc setBuffer:yb offset:0 atIndex:2];
                for (int st = 0; st < NST; ++st) {
                    [enc setBuffer:buf offset:(size_t) st*S atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(800, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double t = cb.GPUEndTime - cb.GPUStartTime;
                if (rep > 0 && t < best) best = t;
            }
            printf("%s  %7.2f us/op  %6.0f GB/s\n", names[k], best/NST*1e6, (double) 10*640*660*NST/best/1e9);
        }
    }
    return 0;
}
