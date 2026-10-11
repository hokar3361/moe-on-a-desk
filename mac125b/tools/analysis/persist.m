// persist: a chain of N dependent bf16 matvecs shaped like the HC mix (down 10240->320, up 320->10240, alternating),
// each op reads the previous op's output. Three executions:
//   sep : one dispatch per op + memory barrier (ggml style)
//   per : one persistent dispatch, grid barrier between ops
//   pf  : persistent, and every simdgroup loads its first weight chunk of op k+1 into registers BEFORE the grid barrier
// B = 1 token. build: clang -fobjc-arc -framework Metal -framework Foundation persist.m -o persist
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;

// weights: op k is [rows_k][K_k] bf16 (row major); down: rows 320, K 10240; up: rows 10240, K 320
// x buffers: xa (len 10240), xb (len 320). down: xa -> xb ; up: xb -> xa

#define NSG 4

struct args_t { uint nops; uint pf; };

// one row-block of a "down" op: 320 rows of K=10240; work item = 1 row per simdgroup, lanes stride K with bf16x8 (16 B) loads
// one row-block of an "up" op: 10240 rows of K=320; work item = 4 rows per simdgroup, 8 lanes per row, 40 elems per lane

static inline float dot_down(device const bfloat * w, device const float * x, ushort lane, thread uint4 * pre, bool have_pre) {
    // K = 10240 = 32 lanes * 40 chunks of 8
    float s = 0.0f;
    for (int c = 0; c < 40; ++c) {
        uint4 q;
        if (have_pre && c < 4) { q = pre[c]; } else { q = ((device const uint4 *) w)[c*32 + lane]; }
        const int k0 = (c*32 + lane)*8;
        const float4 xa = *(device const float4 *) (x + k0), xb4 = *(device const float4 *) (x + k0 + 4);
        s += as_type<float>(q.x << 16)*xa[0] + as_type<float>(q.x & 0xffff0000u)*xa[1]
           + as_type<float>(q.y << 16)*xa[2] + as_type<float>(q.y & 0xffff0000u)*xa[3]
           + as_type<float>(q.z << 16)*xb4[0] + as_type<float>(q.z & 0xffff0000u)*xb4[1]
           + as_type<float>(q.w << 16)*xb4[2] + as_type<float>(q.w & 0xffff0000u)*xb4[3];
    }
    return simd_sum(s);
}

static inline void op_body(device const char * W, device const uint * woff, uint k, device float * xa, device float * xb,
                           uint gsg, uint nsg_total, ushort lane, thread uint4 * pre, bool have_pre) {
    device const bfloat * w = (device const bfloat *) (W + woff[k]);
    if ((k & 1) == 0) {
        // down: 320 rows, row r handled by simdgroup r % nsg_total
        for (uint r = gsg; r < 320; r += nsg_total) {
            const float v = dot_down(w + (size_t) r*10240, xa, lane, pre, have_pre && r == gsg);
            if (lane == 0) xb[r] = v;
        }
    } else {
        // up: 10240 rows of K=320; 4 rows per simdgroup step, 8 lanes per row, each lane 40 elements (5 x 8)
        const ushort lr = lane/8, lc = lane%8;
        for (uint r0 = gsg*4; r0 < 10240; r0 += nsg_total*4) {
            const uint r = r0 + lr;
            device const uint4 * wr = (device const uint4 *) (w + (size_t) r*320);
            float s = 0.0f;
            for (int c = 0; c < 5; ++c) {
                const uint4 q = (have_pre && r0 == gsg*4 && c == 0) ? pre[lr] : wr[c*8 + lc];
                const int k0 = (c*8 + lc)*8;
                const float4 a = *(device const float4 *) (xb + k0), b = *(device const float4 *) (xb + k0 + 4);
                s += as_type<float>(q.x << 16)*a[0] + as_type<float>(q.x & 0xffff0000u)*a[1]
                   + as_type<float>(q.y << 16)*a[2] + as_type<float>(q.y & 0xffff0000u)*a[3]
                   + as_type<float>(q.z << 16)*b[0] + as_type<float>(q.z & 0xffff0000u)*b[1]
                   + as_type<float>(q.w << 16)*b[2] + as_type<float>(q.w & 0xffff0000u)*b[3];
            }
            s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 4);
            if (lc == 0) xa[r] = s*1e-3f;
        }
    }
}

static inline void prefetch_next(device const char * W, device const uint * woff, uint k, uint gsg, ushort lane, thread uint4 * pre) {
    device const bfloat * w = (device const bfloat *) (W + woff[k]);
    if ((k & 1) == 0) {
        device const uint4 * wr = (device const uint4 *) (w + (size_t) gsg*10240);
        for (int c = 0; c < 4; ++c) pre[c] = wr[c*32 + lane];
    } else {
        const ushort lr = lane/8, lc = lane%8;
        device const uint4 * wr = (device const uint4 *) (w + (size_t) (gsg*4 + lr)*320);
        pre[lr] = wr[lc];
    }
}

kernel void k_sep(device const char * W [[buffer(0)]], device const uint * woff [[buffer(1)]], device float * xa [[buffer(2)]],
                  device float * xb [[buffer(3)]], constant uint & k [[buffer(4)]],
                  uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint4 pre[4];
    op_body(W, woff, k, xa, xb, tg*NSG + sg, ntg*NSG, lane, pre, false);
}

kernel void k_per(device const char * W [[buffer(0)]], device const uint * woff [[buffer(1)]], device float * xa [[buffer(2)]],
                  device float * xb [[buffer(3)]], constant args_t & a [[buffer(4)]], device atomic_uint * bar [[buffer(5)]],
                  uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], ushort lid [[thread_index_in_threadgroup]]) {
    device coherent(device) float * cxa = (device coherent(device) float *) xa;
    device coherent(device) float * cxb = (device coherent(device) float *) xb;
    uint4 pre[4];
    bool have = false;
    const uint gsg = tg*NSG + sg, nsg = ntg*NSG;
    for (uint k = 0; k < a.nops; ++k) {
        op_body(W, woff, k, (device float *) cxa, (device float *) cxb, gsg, nsg, lane, pre, have);
        have = false;
        if (a.pf && k + 1 < a.nops) { prefetch_next(W, woff, k + 1, gsg, lane, pre); have = true; }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
            while (atomic_load_explicit(bar, memory_order_relaxed) < (k + 1)*ntg) {}
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
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
        id<MTLComputePipelineState> ps_sep = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_sep"] error:&err];
        id<MTLComputePipelineState> ps_per = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_per"] error:&err];
        if (!ps_sep || !ps_per) { NSLog(@"%@", err); return 1; }
        printf("max threads per tg: per %lu\n", (unsigned long) ps_per.maxTotalThreadsPerThreadgroup);
        const uint N = 64;                        // ops (32 down + 32 up), 64 x 6.25 MB = 400 MB of weights
        const size_t wsz = (size_t) 10240*320*2;
        id<MTLBuffer> W = [dev newBufferWithLength:wsz*N options:MTLResourceStorageModeShared];
        { uint32_t * p = (uint32_t *) W.contents; uint32_t s = 1; for (size_t i = 0; i < wsz*N/4; ++i) { s = s*1664525u + 1013904223u; p[i] = (s & 0x3f7f3f7fu); } }
        id<MTLBuffer> woff = [dev newBufferWithLength:4*N options:MTLResourceStorageModeShared];
        for (uint k = 0; k < N; ++k) ((uint32_t *) woff.contents)[k] = (uint32_t) (wsz*k);
        id<MTLBuffer> xa = [dev newBufferWithLength:4*10240 options:MTLResourceStorageModeShared];
        id<MTLBuffer> xb = [dev newBufferWithLength:4*320 options:MTLResourceStorageModeShared];
        for (int i = 0; i < 10240; ++i) ((float *) xa.contents)[i] = 0.001f*(i % 97);
        id<MTLBuffer> bar = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        int G[3] = {160, 160, 160}; for (int i = 1; i < argc && i <= 3; ++i) G[i-1] = atoi(argv[i]);
        for (int rep = 0; rep < 6; ++rep) {
            // sep
            for (int gi = 0; gi < 3; ++gi) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:ps_sep];
                [enc setBuffer:W offset:0 atIndex:0]; [enc setBuffer:woff offset:0 atIndex:1];
                [enc setBuffer:xa offset:0 atIndex:2]; [enc setBuffer:xb offset:0 atIndex:3];
                for (uint k = 0; k < N; ++k) {
                    [enc setBytes:&k length:4 atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(G[gi], 1, 1) threadsPerThreadgroup:MTLSizeMake(32*4, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (rep >= 2) printf("sep  G=%3d  %.2f us/op\n", G[gi], (cb.GPUEndTime - cb.GPUStartTime)*1e6/N);
            }
            for (uint pf = 0; pf < 2; ++pf) {
                for (int gi = 0; gi < 3; ++gi) {
                    *(uint32_t *) bar.contents = 0;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps_per];
                    [enc setBuffer:W offset:0 atIndex:0]; [enc setBuffer:woff offset:0 atIndex:1];
                    [enc setBuffer:xa offset:0 atIndex:2]; [enc setBuffer:xb offset:0 atIndex:3];
                    uint32_t a[2] = {N, pf};
                    [enc setBytes:a length:8 atIndex:4];
                    [enc setBuffer:bar offset:0 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(G[gi], 1, 1) threadsPerThreadgroup:MTLSizeMake(32*4, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (rep >= 2) printf("%s G=%3d  %.2f us/op\n", pf ? "pf  " : "per ", G[gi], (cb.GPUEndTime - cb.GPUStartTime)*1e6/N);
                }
            }
        }
    }
    return 0;
}
