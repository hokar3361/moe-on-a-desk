// idlebench: GPU time of a fixed ~20 ms workload (a chain of 6.5 MB streaming stages with barriers) when the GPU was idle
// for G ms before it (CPU spinning meanwhile, as the engine does). Shows whether GPU-side DVFS costs time after short idles.
// build: clang -fobjc-arc -framework Metal -framework Foundation idlebench.m -o idlebench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void k_stream(device const uint4 * src [[buffer(0)]], device uint * out [[buffer(1)]], constant uint & n [[buffer(2)]],
                     uint tid [[thread_position_in_grid]], uint ntot [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint i = tid; i < n; i += ntot) acc ^= src[i];
    if (acc.x == 0x12345678u && acc.y == 0x9abcdef0u) out[0] = acc.z;
}
kernel void k_spin(device uint * out [[buffer(0)]], constant uint & iters [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float a = tid;
    for (uint i = 0; i < iters; ++i) a = fma(a, 0.9999f, 0.0001f);
    if (a == 123.456f) out[1] = 1;
}
)";

static double now_ms(void) {
    static mach_timebase_info_data_t tb; if (tb.denom == 0) mach_timebase_info(&tb);
    return (double) mach_absolute_time()*tb.numer/tb.denom/1e6;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLComputePipelineState> ps  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_stream"] error:&err];
        id<MTLComputePipelineState> psp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_spin"] error:&err];
        const int NS = 160;                      // stages
        const size_t SZ = 6500000;               // bytes per stage
        id<MTLBuffer> W = [dev newBufferWithLength:SZ*NS options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        const uint n16 = (uint) (SZ/16);
        const double gaps[] = {0, 0.5, 1.5, 3, 6, 12};
        for (int keep = 0; keep < 2; ++keep) {
            for (int gi = 0; gi < 6; ++gi) {
                double best = 1e9, sum = 0; int cnt = 0;
                for (int rep = 0; rep < 12; ++rep) {
                    // idle gap (CPU spinning); keep=1: a tiny GPU spin kernel runs during the gap
                    const double t0 = now_ms();
                    if (keep && gaps[gi] > 0) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:psp];
                        uint iters = (uint) (gaps[gi]*4000);
                        [enc setBuffer:out offset:0 atIndex:0]; [enc setBytes:&iters length:4 atIndex:1];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit];
                    }
                    while (now_ms() - t0 < gaps[gi]) {}
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:out offset:0 atIndex:1];
                    [enc setBytes:&n16 length:4 atIndex:2];
                    for (int s = 0; s < NS; ++s) {
                        [enc setBuffer:W offset:SZ*s atIndex:0];
                        [enc dispatchThreads:MTLSizeMake(40*1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    [enc endEncoding];
                    [cb commit];
                    while (cb.status != MTLCommandBufferStatusCompleted) {}
                    const double t = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
                    if (rep >= 2) { sum += t; cnt++; if (t < best) best = t; }
                }
                printf("keepalive=%d gap %5.1f ms: GPU time mean %.2f ms, best %.2f ms\n", keep, gaps[gi], sum/cnt, best);
            }
        }
    }
    return 0;
}
