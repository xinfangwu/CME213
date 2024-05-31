#define ARMA_ALLOW_FAKE_GCC
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <algorithm>
#include <armadillo>
#include <cassert>
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < C.n_rows && j < C.n_cols) {
    nn_real sum = 0;
    for (int k = 0; k < A.n_cols; k++) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = alpha * sum + beta * C(i, j);
  }
}

void basicGEMMColumnMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                          nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  dim3 threads(16, 16);
  dim3 grid((C.n_rows + threads.x - 1) / threads.x,
            (C.n_cols + threads.y - 1) / threads.y);
  BasicMatMulColumnMajor<<<grid, threads>>>(A, B, C, alpha, beta);

  check_launch("basicGEMMColumnMajor");
}

__global__ void BasicMatMulRowMajor(DeviceMatrix A, DeviceMatrix B,
                                    DeviceMatrix C, nn_real alpha,
                                    nn_real beta) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < C.n_rows && j < C.n_cols) {
    nn_real sum = 0;
    for (int k = 0; k < A.n_cols; k++) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = alpha * sum + beta * C(i, j);
  }
}

void basicGEMMRowMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                       nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  dim3 threads(16, 16);
  dim3 grid((C.n_cols + threads.x - 1) / threads.x,
            (C.n_rows + threads.y - 1) / threads.y);
  BasicMatMulRowMajor<<<grid, threads>>>(A, B, C, alpha, beta);

  check_launch("basicGEMMRowMajor");
}

template <int blockSizeX, int blockSizeY>
__global__ void SharedMemoryMatMul(DeviceMatrix A, DeviceMatrix B,
                                   DeviceMatrix C, nn_real alpha,
                                   nn_real beta) {

  // row,col of element this thread is responsible for
  int row = threadIdx.y;
  int col = threadIdx.x;

  nn_real sum = 0; // accumulator for the value of the current element
  __shared__ nn_real As[blockSizeX][blockSizeY]; // shared memory for A
  __shared__ nn_real Bs[blockSizeX][blockSizeY]; // shared memory for B
  int numProds = (A.n_cols + blockSizeX - 1) / blockSizeX;

  for (int k = 0; k < numProds; k++) {
    // load the current block of A and B into shared memory
    // x goes along cols for A
    if (k * blockSizeX + col < A.n_cols &&
        row + blockIdx.y * blockSizeY < A.n_rows) {
      As[row][col] = A(row + blockIdx.y * blockSizeY, k * blockSizeX + col);
    } else {
      As[row][col] = 0;
    }
    // x goes along rows for B
    if (k * blockSizeX + row < B.n_rows &&
        col + blockIdx.x * blockSizeY < B.n_cols) {
      Bs[row][col] = B(k * blockSizeX + row, col + blockIdx.x * blockSizeY);
    } else {
      Bs[row][col] = 0;
    }
    __syncthreads();
    // compute the value of the current element
    for (int i = 0; i < blockSizeX; i++) {
      sum += As[row][i] * Bs[i][col];
    }
    __syncthreads();
  }
  // write the value of the current element to C
  int writeRow = row + blockIdx.y * blockSizeY;
  int writeCol = col + blockIdx.x * blockSizeX;
  if (writeRow < C.n_rows && writeCol < C.n_cols) {
    C(writeRow, writeCol) = alpha * sum + beta * C(writeRow, writeCol);
  }
}

void sharedMemoryGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  constexpr int blockSize = 16;
  dim3 threads(blockSize, blockSize);
  dim3 grid((C.n_cols + threads.x - 1) / threads.x,
            (C.n_rows + threads.y - 1) / threads.y);
  SharedMemoryMatMul<blockSize, blockSize>
      <<<grid, threads>>>(A, B, C, alpha, beta);

  check_launch("sharedMemoryGEMM");
}

// 32x32 Hierarchical Tiling
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void TiledMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                            bool transb, DeviceMatrix C, nn_real alpha,
                            nn_real beta) {
  nn_real frag_a[2] = {0, 0};
  nn_real frag_b[4] = {0, 0, 0, 0};
  nn_real accumulator[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

  const int wrap_tile_dimx = 16;
  const int wrap_tile_dimy = 16;
  const int wrap_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int block_tile_dimy = blockItemsM / wrap_tile_dimy;
  const int wx = (wrap_id / block_tile_dimy) * wrap_tile_dimx;
  const int wy = (wrap_id % block_tile_dimy) * wrap_tile_dimy;
  const int tx = lane_id / 8;
  const int ty = lane_id % 8;

  const int K = transa ? A.n_rows : A.n_cols;
  const int A_max_row = transa ? A.n_cols : A.n_rows;
  const int A_max_col = transa ? A.n_rows : A.n_cols;
  const int B_max_row = transb ? B.n_cols : B.n_rows;
  const int B_max_col = transb ? B.n_rows : B.n_cols;

  __shared__ nn_real SMEMA[blockItemsK][blockItemsM + 1];
  __shared__ nn_real SMEMB[blockItemsK][blockItemsN + 1];

  for (int m = 0; m < (K + blockItemsK - 1) / blockItemsK; m++) {
    int by = threadIdx.x % blockItemsM;
    for (int bx = threadIdx.x / blockItemsM; bx < blockItemsK;
         bx += num_thread / blockItemsM) {
      if (blockItemsM * blockIdx.x + by < A_max_row &&
          m * blockItemsK + bx < A_max_col) {
        SMEMA[bx][by] =
            A(blockItemsM * blockIdx.x + by, m * blockItemsK + bx, transa);
      } else {
        SMEMA[bx][by] = 0;
      }
    }
    by = threadIdx.x % blockItemsK;
    for (int bx = threadIdx.x / blockItemsK; bx < blockItemsN;
         bx += num_thread / blockItemsK) {
      if (m * blockItemsK + by < B_max_row &&
          blockItemsN * blockIdx.y + bx < B_max_col) {
        SMEMB[by][bx] =
            B(m * blockItemsK + by, blockItemsN * blockIdx.y + bx, transb);
      } else {
        SMEMB[by][bx] = 0;
      }
    }
    __syncthreads();

    for (int k = 0; k < blockItemsK; k++) {
      int base_y = (wrap_id % block_tile_dimy) * wrap_tile_dimy + ty;
      int base_x = (wrap_id / block_tile_dimy) * wrap_tile_dimx + tx * 4;

      frag_a[0] = SMEMA[k][base_y];
      frag_a[1] = SMEMA[k][base_y + 8];
      for (int i = 0; i < 4; i++) {
        frag_b[i] = SMEMB[k][base_x + i];
        frag_b[i + 4] = SMEMB[k][base_x + i + wrap_tile_dimx / 2];
      }

      for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 4; x++) {
          accumulator[y][x] += frag_a[y] * frag_b[x];
        }
      }
    }
    __syncthreads();
  }

  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 2; y++) {
      const int Cx = blockItemsN * blockIdx.y + wx + tx * 4 + x;
      const int Cy = blockItemsM * blockIdx.x + wy + ty + y * 8;
      if (Cy < C.n_rows && Cx < C.n_cols) {
        C(Cy, Cx) = alpha * accumulator[y][x] + beta * C(Cy, Cx);
      }
    }
  }
}

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

  check_launch("sharedMemoryGEMM");
}
