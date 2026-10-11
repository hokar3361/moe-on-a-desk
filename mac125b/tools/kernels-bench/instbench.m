// instbench: per-instruction throughput on the Apple GPU (M4 Max), as op-per-lane-per-ns over the whole GPU.
// each kernel runs ITER iterations of an unrolled body over 8 independent chains; the result is written so
// the compiler cannot drop it. build: clang -fobjc-arc -framework Metal -framework Foundation instbench.m -o instbench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
#define ITER 4096
constant uint K = 0x9E3779B9u;

// 8 chains x 1 FMA per iteration
kernel void k_fma(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; for (int c = 0; c < 8; ++c) a[c] = tid*0.001f + c;
    const float m = 0.999f, b = 0.001f;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) a[c] = fma(a[c], m, b); }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// 8 chains x (convert u8->f32 + FMA)
kernel void k_cvt(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] += 0x01010101u; a[c] = fma(float(x[c] & 0xffu), 0.5f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// 8 chains x (u8->f32 via uchar4 vector convert: 4 values) + 4 FMA -> per value
kernel void k_cvt4(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float4 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER/4; ++i) { for (int c = 0; c < 8; ++c) { x[c] += 0x01010101u; a[c] = fma(float4(as_type<uchar4>(x[c])), 0.5f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][1] + a[c][2] + a[c][3]; out[tid] = s;
}
// 8 chains x (shift + and + convert + FMA) : extracting a 2-bit field
kernel void k_bitex(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] = fma(float((x[c] >> (2*(i&15))) & 3u), 0.5f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// 8 chains x (and + select + add) : the q2_0 kernel style
kernel void k_sel(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += select(0.0f, 0.5f, bool(x[c] & (1u << (i&31)))); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// 8 chains x (xor sign into float + FMA)
kernel void k_xor(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] = fma(as_type<float>(as_type<uint>(0.5f) ^ (x[c] << (i&31) & 0x80000000u)), 0.5f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// half -> float as FMA operand
kernel void k_h2f(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; half h[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; h[c] = half(tid&7) + half(c); }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { h[c] += half(1); a[c] = fma(float(h[c]), 0.5f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// threadgroup random uint loads (8 independent), table 2 KiB
kernel void k_tgu(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup uint t[512];
    for (int i = lid; i < 512; i += 256) t[i] = i*K;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint x[8]; for (int c = 0; c < 8; ++c) x[c] = tid*K + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = t[(x[c] ^ i) & 511] + x[c]; } }
    uint s = 0; for (int c = 0; c < 8; ++c) s += x[c]; out[tid] = s;
}
// threadgroup random float4 loads (8 independent), table 8 KiB
kernel void k_tgf4(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup float4 t[512];
    for (int i = lid; i < 512; i += 256) t[i] = float4(i, i+1, i+2, i+3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c]*1664525u + 1013904223u; a[c] += t[(x[c] >> 16) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// same as k_tgf4 without the load (cost of the index arithmetic + add)
kernel void k_tgf4_base(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float4 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c]*1664525u + 1013904223u; a[c] += float4(as_type<float>((x[c] >> 16) & 511u)); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// threadgroup random uint2 loads (8 bytes: one iq2_xxs grid entry)
kernel void k_tgu2(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 t[256];
    for (int i = lid; i < 256; i += 256) t[i] = uint2(i*K, i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c]*1664525u + 1013904223u; a[c] += t[(x[c] >> 16) & 255]; } }
    uint s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][1]; out[tid] = s;
}
// device (L1) random float4 loads from an 8 KiB table
kernel void k_devf4(device float * out [[buffer(0)]], device const float4 * t [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float4 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c]*1664525u + 1013904223u; a[c] += t[(x[c] >> 16) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// constant-space random float4 loads from an 8 KiB table
constant float4 ctab[512] = { float4(1) };
kernel void k_cstf4(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float4 a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c]*1664525u + 1013904223u; a[c] += ctab[(x[c] >> 16) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}

// single-op chains: 8 chains x 1 op per iteration (the loop counter is shared)
kernel void k_iadd(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    uint x[8]; for (int c = 0; c < 8; ++c) x[c] = tid*K + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) x[c] = x[c] + 0x01234567u; }
    uint s = 0; for (int c = 0; c < 8; ++c) s ^= x[c]; out[tid] = s;
}
kernel void k_ixor(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    uint x[8]; for (int c = 0; c < 8; ++c) x[c] = tid*K + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) x[c] = (x[c] ^ 0x01234567u) + 1u; }
    uint s = 0; for (int c = 0; c < 8; ++c) s ^= x[c]; out[tid] = s;
}
kernel void k_shlc(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    uint x[8]; for (int c = 0; c < 8; ++c) x[c] = tid*K + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) x[c] = (x[c] << 3) + 1u; }
    uint s = 0; for (int c = 0; c < 8; ++c) s ^= x[c]; out[tid] = s;
}
kernel void k_bfe(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    uint x[8]; for (int c = 0; c < 8; ++c) x[c] = tid*K + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) x[c] = ((x[c] >> 5) & 0x3fu) + x[c]; }
    uint s = 0; for (int c = 0; c < 8; ++c) s ^= x[c]; out[tid] = s;
}
kernel void k_cvtonly(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) a[c] = float(x[c] + i) + a[c]; }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
kernel void k_fadd(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; for (int c = 0; c < 8; ++c) a[c] = tid*0.001f + c;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) a[c] = a[c] + 1.5f; }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
kernel void k_hfma(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    half a[8]; for (int c = 0; c < 8; ++c) a[c] = half(c);
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) a[c] = fma(a[c], half(0.999h), half(0.001h)); }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// magic half2: 2 values per (and + or) then 2 half->float fma
kernel void k_magic(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = 0; x[c] = tid*K + c; }
    for (int i = 0; i < ITER/2; ++i) { for (int c = 0; c < 8; ++c) { x[c] += 0x00010001u; const half2 h = as_type<half2>((x[c] & 0x00ff00ffu) | 0x64006400u); a[c] = fma(float(h[0]), 0.5f, a[c]); a[c] = fma(float(h[1]), 0.25f, a[c]); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// mixed: 1 int op + 1 fma per iteration in separate chains (dual issue?)
kernel void k_mix(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[8]; uint x[8]; for (int c = 0; c < 8; ++c) { a[c] = c; x[c] = tid*K + c; }
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { x[c] = x[c] + 0x01234567u; a[c] = fma(a[c], 0.999f, 0.001f); } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c] + x[c]; out[tid] = s;
}

// threadgroup float4 reads, all lanes of the simdgroup read the same address (broadcast), address walks
kernel void k_tgf4_bc(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup float4 t[512];
    for (int i = lid; i < 512; i += 256) t[i] = float4(i, i+1, i+2, i+3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 a[8]; for (int c = 0; c < 8; ++c) a[c] = 0;
    const uint sg = lid/32;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += t[(i*8 + c + sg) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// threadgroup float4 reads, lane-consecutive addresses
kernel void k_tgf4_seq(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup float4 t[512];
    for (int i = lid; i < 512; i += 256) t[i] = float4(i, i+1, i+2, i+3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 a[8]; for (int c = 0; c < 8; ++c) a[c] = 0;
    const uint ln = lid%32;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += t[(ln + 32*((i + c) & 7)) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// threadgroup float reads, lane-consecutive
kernel void k_tgf_seq(device float * out [[buffer(0)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup float t[2048];
    for (int i = lid; i < 2048; i += 256) t[i] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a[8]; for (int c = 0; c < 8; ++c) a[c] = 0;
    const uint ln = lid%32;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += t[(ln + 32*((i + c) & 63)) & 2047]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c]; out[tid] = s;
}
// device float4 reads, all lanes same address (L1 broadcast)
kernel void k_devf4_bc(device float * out [[buffer(0)]], device const float4 * t [[buffer(1)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    float4 a[8]; for (int c = 0; c < 8; ++c) a[c] = 0;
    const uint sg = lid/32;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += t[(i*8 + c + sg) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
// device float4 reads, lane-consecutive (L1)
kernel void k_devf4_seq(device float * out [[buffer(0)]], device const float4 * t [[buffer(1)]], uint tid [[thread_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) {
    float4 a[8]; for (int c = 0; c < 8; ++c) a[c] = 0;
    const uint ln = lid%32;
    for (int i = 0; i < ITER; ++i) { for (int c = 0; c < 8; ++c) { a[c] += t[(ln + 32*((i + c) & 7)) & 511]; } }
    float s = 0; for (int c = 0; c < 8; ++c) s += a[c][0] + a[c][3]; out[tid] = s;
}
)";

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const NSUInteger nthreads = 40*1024*8; // plenty of threads for 40 cores
        id<MTLBuffer> out = [dev newBufferWithLength:nthreads*16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> tab = [dev newBufferWithLength:512*16 options:MTLResourceStorageModeShared];
        // ops per thread per kernel (the measured op count)
        NSArray * names = @[@"k_fma", @"k_tgf4", @"k_tgf4_bc", @"k_tgf4_seq", @"k_tgf_seq", @"k_devf4", @"k_devf4_bc", @"k_devf4_seq"];
        const double per[] = { 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0, 8*4096.0 };
        for (NSUInteger k = 0; k < names.count; ++k) {
            id<MTLFunction> f = [lib newFunctionWithName:names[k]];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!ps) { NSLog(@"%@ %@", names[k], err); continue; }
            double best = 1e30;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:out offset:0 atIndex:0];
                [enc setBuffer:tab offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double t = cb.GPUEndTime - cb.GPUStartTime;
                if (rep > 0 && t < best) best = t;
            }
            const double ops = per[k]*nthreads;
            printf("%-12s %8.3f ms  %7.2f Gop/s  (%.2f ns per op per 1k lanes)\n", [names[k] UTF8String], best*1e3, ops/best/1e9, best/ops*1e9*1000);
        }
    }
    return 0;
}
