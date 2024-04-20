#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "gtest/gtest.h"
#include "utils.h"

using std::vector;

int N = 32;

class dvector
{
private:
    int n = 0;
    int *data = nullptr;

public:
    dvector() = default;
    dvector(const int n_)
    {
        assert(n_ > 0);
        checkCudaErrors(cudaMalloc(&data, sizeof(int) * n_));
        n = n_;
    }
    ~dvector()
    {
        if (data)
        {
            checkCudaErrors(cudaFree(data));
            n = 0;
        }
    }
    int *raw_pointer() { return data; }
};

__device__ __host__ int f(int i)
{
    return i * (i % 10);
}

__global__ void kernel(int n, int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = f(i);
}

TEST(CUDA, assign_grid)
{
    dvector d_output(N);

    dim3 block_dim(1024);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);
    kernel<<<grid_dim, block_dim>>>(N, d_output.raw_pointer());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    vector<int> h_output(N);
    checkCudaErrors(cudaMemcpy(&h_output[0], d_output.raw_pointer(),
                               sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i)
        ASSERT_EQ(h_output[i], f(i));
}

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, argv, "N"))
    {
        N = getCmdLineArgumentInt(argc, argv, "N");
        assert(N >= 0);
        printf("Using %d threads = %d warps\n", N, (N + 31) / 32);
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}