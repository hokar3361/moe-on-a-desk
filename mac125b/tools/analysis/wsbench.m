// wsbench: chain of N dependent bf16 matvecs shaped like the HC mix (down 10240->320, up 320->10240 alternating), one
// dispatch per op with a memory barrier between ops (ggml style). Two work distributions:
//   static : grid of G threadgroups, rows statically split (as the stock kernels)
//   steal  : G threadgroups that pull row-blocks from an atomic counter until the op is done (no inter-group waiting,
//            so no deadlock), every group finishes within one block of the others -> short tail
// build: clang -fobjc-arc -framework Metal -framework Foundation wsbench.m -o wsbench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
#define NSG 4

// down: rows of K=10240 (one row per simdgroup); up: rows of K=320 (4 rows per simdgroup, 8 lanes per row)
static inline void rows_down(device const bfloat * w, device const float * x, device float * y, uint r, ushort lane) {
    device const uint4 * wr = (device const uint4 *) (w + (size_t) r*10240);
    float s = 0.0f;
    for (int c = 0; c < 40; ++c) {
        const uint4 q = wr[c*32 + lane];
        const int k0 = (c*32 + lane)*8;
        const float4 a = *(device const float4 *) (x + k0), b = *(device const float4 *) (x + k0 + 4);
        s += as_type<float>(q.x << 16)*a[0] + as_type<float>(q.x & 0xffff0000u)*a[1]
           + as_type<float>(q.y << 16)*a[2] + as_type<float>(q.y & 0xffff0000u)*a[3]
           + as_type<float>(q.z << 16)*b[0] + as_type<float>(q.z & 0xffff0000u)*b[1]
           + as_type<float>(q.w << 16)*b[2] + as_type<float>(q.w & 0xffff0000u)*b[3];
    }
    s = simd_sum(s);
    if (lane == 0) y[r] = s;
}
static inline void rows_up(device const bfloat * w, device const float * x, device float * y, uint r0, ushort lane) {
    const ushort lr = lane/8, lc = lane%8;
    const uint r = r0 + lr;
    device const uint4 * wr = (device const uint4 *) (w + (size_t) r*320);
    float s = 0.0f;
    for (int c = 0; c < 5; ++c) {
        const uint4 q = wr[c*8 + lc];
        const int k0 = (c*8 + lc)*8;
        const float4 a = *(device const float4 *) (x + k0), b = *(device const float4 *) (x + k0 + 4);
        s += as_type<float>(q.x << 16)*a[0] + as_type<float>(q.x & 0xffff0000u)*a[1]
           + as_type<float>(q.y << 16)*a[2] + as_type<float>(q.y & 0xffff0000u)*a[3]
           + as_type<float>(q.z << 16)*b[0] + as_type<float>(q.z & 0xffff0000u)*b[1]
           + as_type<float>(q.w << 16)*b[2] + as_type<float>(q.w & 0xffff0000u)*b[3];
    }
    s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 4);
    if (lc == 0) y[r] = s*1e-3f;
}

kernel void k_static(device const char * W [[buffer(0)]], device float * xa [[buffer(1)]], device float * xb [[buffer(2)]],
                     constant uint & k [[buffer(3)]],
                     uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const bfloat * w = (device const bfloat *) (W + (size_t) k*10240*320*2);
    const uint gsg = tg*NSG + sg, nsg = ntg*NSG;
    if ((k & 1) == 0) { for (uint r = gsg; r < 320; r += nsg) rows_down(w, xa, xb, r, lane); }
    else              { for (uint r0 = gsg*4; r0 < 10240; r0 += nsg*4) rows_up(w, xb, xa, r0, lane); }
}

// unit = UNIT simdgroup-steps; down: 1 row per simdgroup-step, up: 4 rows
kernel void k_steal(device const char * W [[buffer(0)]], device float * xa [[buffer(1)]], device float * xb [[buffer(2)]],
                    constant uint & k [[buffer(3)]], device atomic_uint * ctr [[buffer(4)]], constant uint & unit [[buffer(5)]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const bfloat * w = (device const bfloat *) (W + (size_t) k*10240*320*2);
    const uint nsteps = (k & 1) == 0 ? 320 : 10240/4;
    while (true) {
        uint s0 = 0;
        if (lane == 0) s0 = atomic_fetch_add_explicit(ctr + k, unit, memory_order_relaxed);
        s0 = simd_broadcast(s0, 0);
        if (s0 >= nsteps) break;
        const uint s1 = min(s0 + unit, nsteps);
        for (uint s = s0; s < s1; ++s) {
            if ((k & 1) == 0) rows_down(w, xa, xb, s, lane);
            else              rows_up(w, xb, xa, s*4, lane);
        }
    }
}
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLComputePipelineState> ps_st = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_static"] error:&err];
        id<MTLComputePipelineState> ps_ws = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_steal"] error:&err];
        const uint N = 64;
        const size_t wsz = (size_t) 10240*320*2;
        id<MTLBuffer> W = [dev newBufferWithLength:wsz*N options:MTLResourceStorageModeShared];
        { uint32_t * p = (uint32_t *) W.contents; uint32_t s = 1; for (size_t i = 0; i < wsz*N/4; ++i) { s = s*1664525u + 1013904223u; p[i] = (s & 0x3f7f3f7fu); } }
        id<MTLBuffer> xa = [dev newBufferWithLength:4*10240 options:MTLResourceStorageModeShared];
        id<MTLBuffer> xb = [dev newBufferWithLength:4*320 options:MTLResourceStorageModeShared];
        for (int i = 0; i < 10240; ++i) ((float *) xa.contents)[i] = 0.001f*(i % 97);
        id<MTLBuffer> ctr = [dev newBufferWithLength:4*N options:MTLResourceStorageModeShared];
        const int Gs[4] = {160, 320, 640, 1280};
        const int Gw[4] = {80, 160, 240, 320};
        const uint units[3] = {1, 2, 4};
        for (int rep = 0; rep < 5; ++rep) {
            for (int gi = 0; gi < 4; ++gi) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:ps_st];
                [enc setBuffer:W offset:0 atIndex:0]; [enc setBuffer:xa offset:0 atIndex:1]; [enc setBuffer:xb offset:0 atIndex:2];
                for (uint k = 0; k < N; ++k) {
                    [enc setBytes:&k length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(Gs[gi], 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (rep >= 1) printf("static G=%4d          %.2f us/op\n", Gs[gi], (cb.GPUEndTime - cb.GPUStartTime)*1e6/N);
            }
            for (int ui = 0; ui < 3; ++ui) {
                for (int gi = 0; gi < 4; ++gi) {
                    memset(ctr.contents, 0, 4*N);
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [enc setComputePipelineState:ps_ws];
                    [enc setBuffer:W offset:0 atIndex:0]; [enc setBuffer:xa offset:0 atIndex:1]; [enc setBuffer:xb offset:0 atIndex:2];
                    [enc setBuffer:ctr offset:0 atIndex:4];
                    [enc setBytes:&units[ui] length:4 atIndex:5];
                    for (uint k = 0; k < N; ++k) {
                        [enc setBytes:&k length:4 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(Gw[gi], 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (rep >= 1) printf("steal  G=%4d unit=%u  %.2f us/op\n", Gw[gi], units[ui], (cb.GPUEndTime - cb.GPUStartTime)*1e6/N);
                }
            }
        }
    }
    return 0;
}
