// chainbench: cost per stage of a chain of N dependent small elementwise ops (y = f(x) over E floats), three ways:
//   sep : one dispatch per op, memory barrier between them (what ggml does)
//   tg1 : one dispatch, ONE threadgroup of 1024 threads runs all ops, threadgroup_barrier(mem_device) between ops
//   grid: one dispatch, G co-resident threadgroups, device-wide atomic barrier between ops (persistent)
// build: clang -fobjc-arc -framework Metal -framework Foundation chainbench.m -o chainbench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString * src = @R"(
#include <metal_stdlib>
using namespace metal;

// op k: buf[(k+1)%2] = buf[k%2]*a + b, then sigmoid-ish nonlinearity (cheap, dependent)
static inline float f(float v, int k) { return v*0.999f + 0.001f*(float)(k & 7); }

kernel void k_sep(device float * a [[buffer(0)]], device float * b [[buffer(1)]], constant int & E [[buffer(2)]], constant int & k [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
    if ((int) tid < E) b[tid] = f(a[tid], k);
}

kernel void k_tg1(device float * a [[buffer(0)]], device float * b [[buffer(1)]], constant int & E [[buffer(2)]], constant int & N [[buffer(3)]],
                  ushort lid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
    for (int k = 0; k < N; ++k) {
        device float * x = (k & 1) ? b : a;
        device float * y = (k & 1) ? a : b;
        for (int i = lid; i < E; i += ntg) y[i] = f(x[i], k);
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void k_grid(device float * a [[buffer(0)]], device float * b [[buffer(1)]], constant int & E [[buffer(2)]], constant int & N [[buffer(3)]],
                   device atomic_uint * bar [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]], uint ntgs [[threadgroups_per_grid]],
                   ushort lid [[thread_index_in_threadgroup]], ushort ntg [[threads_per_threadgroup]]) {
    device coherent(device) float * ca = (device coherent(device) float *) a;
    device coherent(device) float * cb = (device coherent(device) float *) b;
    const uint stride = ntgs*ntg;
    for (int k = 0; k < N; ++k) {
        device coherent(device) float * x = (k & 1) ? cb : ca;
        device coherent(device) float * y = (k & 1) ? ca : cb;
        for (uint i = tg*ntg + lid; i < (uint) E; i += stride) y[i] = f(x[i], k);
        // device-wide barrier: counter reaches (k+1)*ntgs
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
            while (atomic_load_explicit(bar, memory_order_relaxed) < (uint) (k + 1)*ntgs) {}
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
        id<MTLComputePipelineState> ps_sep  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_sep"]  error:&err];
        id<MTLComputePipelineState> ps_tg1  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tg1"]  error:&err];
        id<MTLComputePipelineState> ps_grid = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_grid"] error:&err];
        const int N = 64;
        id<MTLBuffer> A = [dev newBufferWithLength:4*200000 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> Bf = [dev newBufferWithLength:4*200000 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bar = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        // warm the GPU clock with a busy prefix in every command buffer (kernels run L times back to back)
        const int L = 20;
        const int Es[5] = {2560, 7680, 10240*3, 10240*8, 81920*2};
        for (int ie = 0; ie < 5; ++ie) { const int E = Es[ie];
            double best[4] = {1e30, 1e30, 1e30, 1e30};
            for (int rep = 0; rep < 8; ++rep) {
                for (int mode = 0; mode < 4; ++mode) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    for (int l = 0; l < L; ++l) {
                        if (mode == 0) {
                            [enc setComputePipelineState:ps_sep];
                            for (int k = 0; k < N; ++k) {
                                [enc setBuffer:(k & 1) ? Bf : A offset:0 atIndex:0];
                                [enc setBuffer:(k & 1) ? A : Bf offset:0 atIndex:1];
                                [enc setBytes:&E length:4 atIndex:2];
                                [enc setBytes:&k length:4 atIndex:3];
                                [enc dispatchThreads:MTLSizeMake(E, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            }
                        } else if (mode == 1) {
                            [enc setComputePipelineState:ps_tg1];
                            [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Bf offset:0 atIndex:1];
                            [enc setBytes:&E length:4 atIndex:2]; [enc setBytes:&N length:4 atIndex:3];
                            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        } else {
                            const int G = mode == 2 ? 16 : 40;
                            uint32_t z = 0;
                            // reset the counter with a tiny blit-free trick: a separate buffer offset per l
                            [enc setComputePipelineState:ps_grid];
                            [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Bf offset:0 atIndex:1];
                            [enc setBytes:&E length:4 atIndex:2]; [enc setBytes:&N length:4 atIndex:3];
                            [enc setBuffer:bar offset:0 atIndex:4];
                            (void) z;
                            [enc dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            break; // one per command buffer (the counter is reset on the CPU)
                        }
                    }
                    [enc endEncoding];
                    *(uint32_t *) bar.contents = 0;
                    [cb commit];
                    [cb waitUntilCompleted];
                    const double t = cb.GPUEndTime - cb.GPUStartTime;
                    const double per = t/((mode >= 2 ? 1 : L)*(double) N);
                    if (rep > 1 && per < best[mode]) best[mode] = per;
                }
            }
            printf("E=%6d  us/stage: sep %.2f  tg1 %.2f  grid16 %.2f  grid40 %.2f\n", E, best[0]*1e6, best[1]*1e6, best[2]*1e6, best[3]*1e6);
        }
    }
    return 0;
}
