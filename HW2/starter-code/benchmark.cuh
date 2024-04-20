#ifndef _BENCHMARK_CUH
#define _BENCHMARK_CUH

#include "util.cuh"

// Kernel for the benchmark
__global__ void elementwise_add(const int *x, const int *y,
                                int *z, unsigned int stride,
                                unsigned int size) {
    // V TODO: elementwise_add should compute
    // z[i * stride] = x[i * stride] + y[i * stride]
    // where i goes from 0 to size-1.
    // Distribute the work across all CUDA threads allocated by
    // elementwise_add<<<72, 1024>>>(x, y, z, stride, N);
    // Use the CUDA variables gridDim, blockDim, blockIdx, and threadIdx.
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_thread = gridDim.x * blockDim.x;

    for(size_t i=thread_idx; i<size; i+=total_thread){
        z[i*stride] = x[i*stride] + y[i*stride];
    }

}

#endif
