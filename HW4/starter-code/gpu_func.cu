#define ARMA_ALLOW_FAKE_GCC
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <memory>

#include "gpu_func.h"
#include "util.cuh"

__global__ void Warmup() {}

void DWarmup() { Warmup<<<1, 1>>>(); }

/**
 * DeviceAllocator and DeviceMatrix
 */

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n) {
  assert(n >= 0);
  assert(cpu_data != nullptr);
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
  cudaMemcpy(data, cpu_data, nbytes, cudaMemcpyHostToDevice);
}

DeviceAllocator::DeviceAllocator(int n) {
  assert(n >= 0);
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
}

DeviceAllocator::~DeviceAllocator() {
  if (data != nullptr)
    cudaFree(data);
}

int DeviceAllocator::total_bytes() { return nbytes; }

nn_real *DeviceAllocator::memptr() { return data; }

void DeviceAllocator::to_cpu(nn_real *cpu_data) {
  assert(data != nullptr && cpu_data != nullptr);
  cudaMemcpy(cpu_data, data, nbytes, cudaMemcpyDeviceToHost);
}

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols) {
  assert(n_rows >= 0 && n_cols >= 0);
  this->allocator = std::make_shared<DeviceAllocator>(n_rows * n_cols);
  this->data = this->allocator->memptr();
  this->n_rows = n_rows;
  this->n_cols = n_cols;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat) {
  this->allocator = std::make_shared<DeviceAllocator>(
      cpu_mat.memptr(), cpu_mat.n_rows * cpu_mat.n_cols);
  this->data = this->allocator->memptr();
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
}

int DeviceMatrix::total_bytes() { return allocator->total_bytes(); }

nn_real *DeviceMatrix::memptr() { return data; }

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat) {
  allocator->to_cpu(cpu_mat.memptr());
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col, bool transpose) {
  assert(data != nullptr && row >= 0 && row < n_rows && col >= 0 &&
         col < n_cols);
  return transpose ? data[row * n_cols + col] : data[col * n_rows + row];
}
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                           GEMM kernels                           //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

__global__ void BasicMatMulColumnMajor(DeviceMatrix A, DeviceMatrix B,
                                       DeviceMatrix C, nn_real alpha,
                                       nn_real beta) {
  // V TODO: Implement this kernel
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  

  if(col < C.n_cols && row < C.n_rows){
    nn_real sum = 0;
    for(int k =0; k< A.n_cols; k++){
      sum += A(row, k) * B(k, col);
    }
    C(row, col) = alpha * sum + beta * C(row, col);
  }
}

void basicGEMMColumnMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                          nn_real alpha, nn_real beta) {
  // V TODO: Implement this kernel wrapper
  // Remember that column major means that consecutive threads compute
  // consecutive elements in a column of the output matrix

  // check_launch("basicGEMMColumnMajor");
  int numThread_row = 32;
  int numThread_col = 32;
  int numBlock_row = (C.n_rows + numThread_row - 1)/numThread_row;
  int numBlock_col = (C.n_cols + numThread_col - 1)/numThread_col;
  dim3 blockSize(numThread_row, numThread_col);
  dim3 gridSize(numBlock_row, numBlock_col);

  // Launch the kernel
  BasicMatMulColumnMajor<<<gridSize, blockSize>>>(A, B, C, alpha, beta);

  // Check for errors in kernel launch or during execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void BasicMatMulRowMajor(DeviceMatrix A, DeviceMatrix B,
                                    DeviceMatrix C, nn_real alpha,
                                    nn_real beta) {
  // V TODO: Implement this kernel
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(col < C.n_cols && row < C.n_rows){
    nn_real sum = 0;
    for(int k =0; k< A.n_cols; k++){
      sum += A(row, k) * B(k, col);
    }
    C(row, col) = alpha * sum + beta * C(row, col);
  }

}

void basicGEMMRowMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                       nn_real alpha, nn_real beta) {
  // V TODO: Implement this kernel wrapper
  // Remember that row major means that consecutive threads compute
  // consecutive elements in a row of the output matrix

  // check_launch("basicGEMMRowMajor");
  int numThread_row = 128;
  int numThread_col = 8;
  int numBlock_row = (C.n_rows + numThread_row - 1)/numThread_row;
  int numBlock_col = (C.n_cols + numThread_col - 1)/numThread_col;
  dim3 blockSize(numThread_col, numThread_row);
  dim3 gridSize(numBlock_col, numBlock_row);

  // Launch the kernel
  BasicMatMulRowMajor<<<gridSize, blockSize>>>(A, B, C, alpha, beta);

  // Check for errors in kernel launch or during execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

}

#define blockSize_x 32
#define blockSize_y 32

template <int blockSizeX, int blockSizeY>
__global__ void SharedMemoryMatMul(DeviceMatrix A, DeviceMatrix B,
                                   DeviceMatrix C, nn_real alpha,
                                   nn_real beta) {

  // V TODO: Implement this kernel
  // x-> col, y -> row
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ nn_real As[blockSizeX][blockSizeY];
  __shared__ nn_real Bs[blockSizeX][blockSizeY];
  nn_real Cvalue = 0;

  // loop over all sub-matrices of A and B along ->
  for(int m=0; m<((A.n_cols + blockSizeX - 1)/blockSizeX); m++){
    
    // load to shared As
    // one thread one element 
    int in_col = m * blockSizeX + threadIdx.x;
    if(in_col < A.n_cols && row < A.n_rows){
      As[threadIdx.y][threadIdx.x] = A(row, in_col);
    }
    else{
      As[threadIdx.y][threadIdx.x] = 0;
    }

    // load to shared Bs
    // one thread one element 
    int in_row = m * blockSizeY + threadIdx.y;
    if(in_row < B.n_rows && col < B.n_cols){
      Bs[threadIdx.y][threadIdx.x] = B(in_row, col);
    }
    else{
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // do sum
    for (int e=0; e<blockSizeY; e++){
      Cvalue += As[threadIdx.y][e] * Bs[e][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < C.n_rows && col < C.n_cols){
    C(row, col) = alpha * Cvalue + beta * C(row, col);
  }
}

void sharedMemoryGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta) {
  // V TODO: Implement this wrapper
  // check_launch("sharedMemoryGEMM");
  // x -> col, y -> row
  int numThread_x = blockSize_x;
  int numThread_y = blockSize_y;
  int numBlock_x = (C.n_cols + numThread_x - 1)/numThread_x;
  int numBlock_y = (C.n_rows + numThread_y - 1)/numThread_y;
  dim3 blockSize(numThread_x, numThread_y);
  dim3 gridSize(numBlock_x, numBlock_y);

  // Launch the kernel
  SharedMemoryMatMul<blockSize_x, blockSize_y><<<gridSize, blockSize>>>(A, B, C, alpha, beta);

  // Check for errors in kernel launch or during execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

// 32x32 Hierarchical Tiling
// num_thread: number of threads per block
// blockItemsM: number of rows of A in each submatrix of A
// blockItemsN: number of columns of B in each submatrix of B
// blockItemsK: number of columns in submatrix of A and rows in submatrix of B
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void TiledMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                            bool transb, DeviceMatrix C, nn_real alpha,
                            nn_real beta) {
  // TODO: Implement this kernel
  // x-> col, y -> row
  // int row = blockDim.y * blockIdx.y + threadIdx.y;
  // int col = blockDim.x * blockIdx.x + threadIdx.x;
  // __shared__ nn_real As[blockItemsM][blockItemsK];
  // __shared__ nn_real Bs[blockItemsK][blockItemsN];
  // nn_real Cvalue = 0;

  // int K = A.n_cols;

  // // First, Load A tile and B tile into shared memory 
  // // Matrix -> Tile
  // for(int kblock=0; kblock<((K + blockItemsK -1)/blockItemsK); kblock++){

  //   int in_col = kblock * blockItemsK + threadIdx.x;
  //   if(in_col < A.n_cols && row < A.n_rows){
  //     As[threadIdx.y][threadIdx.x] = A(row, in_col);
  //   }
  //   else{
  //     As[threadIdx.y][threadIdx.x] = 0;
  //   }

  //   // load to shared Bs
  //   // one thread one element 
  //   int in_row = kblock * blockItemsK + threadIdx.y;
  //   if(in_row < B.n_rows && col < B.n_cols){
  //     Bs[threadIdx.y][threadIdx.x] = B(in_row, col);
  //   }
  //   else{
  //     Bs[threadIdx.y][threadIdx.x] = 0;
  //   }

  //   __syncthreads();

  //   // Second, Tile -> Warp 
  //   int WarpSizeY = 16;
  //   int WarpSizeX = 8;
  //   int warpRow = threadIdx.y / WarpSizeY;
  //   int warpCol = threadIdx.x / WarpSizeX;
  //   // int laneIndex = threadIdx.y % WarpSize * WarpSize + threadIdx.x % WarpSize;

  //   #pragma unroll
  //   for (int k = 0; k < blockItemsK; k++) {
  //       nn_real frag_a = As[warpRow * WarpSize + laneIndex / ThreadItemsX][k];
  //       nn_real frag_b = Bs[k][warpCol * WarpSize + laneIndex % ThreadItemsX];
  //       #pragma unroll
  //       for (int thread_x = 0; thread_x < ThreadItemsX; ++thread_x) {
  //           #pragma unroll
  //           for (int thread_y = 0; thread_y < ThreadItemsY; ++thread_y) {
  //               Cvalue += frag_a[y] * frag_b[x];
  //           }
  //       }
  //   }
  //   __syncthreads();

  // }

  // if (row < C.n_rows && col < C.n_cols){
  //   C(row, col) = alpha * Cvalue + beta * C(row, col);
  // }
  // check_launch("tiledGEMM");
}


// wrapper for MatMulTile_32_32
void tiledGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C, nn_real alpha,
               nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  constexpr int block_m = 32;
  constexpr int block_n = 32;
  constexpr int block_k = 32;
  constexpr int num_thread = 128;
  dim3 grid((C.n_rows + block_m - 1) / block_m,
            (C.n_cols + block_n - 1) / block_n);
  TiledMatMul<num_thread, block_m, block_n, block_k>
      <<<grid, num_thread>>>(A, false, B, false, C, alpha, beta);

  check_launch("tiledGEMM");
}
