#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <fcntl.h>
__global__ void rd(const unsigned char * p, unsigned int * out) { out[0] = p[0] + p[4096 * 3]; }
int main(int argc, char ** argv) {
    int v = -1; cudaDeviceGetAttribute(&v, cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
    int u = -1; cudaDeviceGetAttribute(&u, cudaDevAttrUnifiedAddressing, 0);
    printf("canUseHostPointerForRegisteredMem %d unifiedAddressing %d\n", v, u);
    int fd = open(argv[1], O_RDONLY);
    unsigned char * m = (unsigned char *) mmap(nullptr, 1 << 24, PROT_READ, MAP_SHARED, fd, 0);
    volatile unsigned s = m[0] + m[4096 * 3];
    cudaError_t e = cudaHostRegister(m, 1 << 24, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    printf("register (portable|readonly): %s\n", cudaGetErrorString(e));
    void * dp = nullptr; e = cudaHostGetDevicePointer(&dp, m, 0);
    printf("device pointer: %s, same as host: %d\n", cudaGetErrorString(e), dp == (void *) m);
    unsigned int * out; cudaMalloc(&out, 4);
    rd<<<1, 1>>>(m, out); e = cudaDeviceSynchronize();
    unsigned int r = 0; cudaMemcpy(&r, out, 4, cudaMemcpyDeviceToHost);
    printf("kernel read via host pointer: %s, value %u expected %u\n", cudaGetErrorString(e), r, (unsigned) s);
}
