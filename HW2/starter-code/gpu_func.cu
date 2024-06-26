#define ARMA_ALLOW_FAKE_GCC
#include <algorithm>
#include <armadillo>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <memory>

#include "gpu_func.h"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                          DeviceAllocator 						//
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n)
{
  // V TODO: implement this constructor
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
  cudaMemcpy(data, cpu_data, nbytes, cudaMemcpyHostToDevice);
}

DeviceAllocator::DeviceAllocator(int n)
{
  // V TODO: implement this constructor
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
}

DeviceAllocator::~DeviceAllocator()
{
  // V TODO: implement this destructor
  cudaFree(data);
  nbytes = 0;
}

void DeviceAllocator::to_cpu(nn_real *cpu_data)
{
  // V TODO: implement this function
  cudaMemcpy(cpu_data, data, nbytes, cudaMemcpyDeviceToHost);
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                          DeviceMatrix 							//
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols)
{
  // V TODO: implement this constructor
  this->n_rows = n_rows;
  this->n_cols = n_cols;
  allocator = std::make_shared<DeviceAllocator>(this->n_rows * this->n_cols);
  data = allocator->data;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat)
{
  // V TODO: implement this constructor
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
  allocator = std::make_shared<DeviceAllocator>(cpu_mat.memptr(), this->n_rows * this->n_cols);
  data = allocator->data;
}

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat)
{
  this->allocator->to_cpu(cpu_mat.memptr());
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col)
{
  // Note that arma matrices are column-major
  return data[col * this->n_rows + row];
}

int DeviceMatrix::total_bytes()
{
  return allocator->nbytes;
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                           CUDA kernels                           //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

/**
 * A CUDA kernel function that applies the sigmoid function element-wise to a
 * matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this kernel function
  // Hint: Use Exp() from common.h
  int row_init = blockIdx.x * blockDim.x + threadIdx.x;
  int col_init = blockIdx.y * blockDim.y + threadIdx.y;
  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;

  for(int row = row_init; row < src.n_rows; row += total_threads_x){
    for(int col = col_init; col < src.n_cols; col += total_threads_y){
        dst(row, col) = 1 / (1 + Exp(-src(row, col)));
    }
  }
  
}

/**
 * A CUDA kernel function that repeats each column of the source matrix `repeat`
 * times and stores the result in the destination matrix.
 *
 * @param src The source matrix to repeat.
 * @param dst The destination matrix to store the repeated columns.
 * @param repeat The number of times to repeat each column.
 */
__global__ void MatRepeatColVec(DeviceMatrix src, DeviceMatrix dst,
                                int repeat)
{
  // TODO: implement this kernel function
  int row_init = blockIdx.x * blockDim.x + threadIdx.x;
  int col_init = blockIdx.y * blockDim.y + threadIdx.y;
  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;

  for(int row = row_init; row < src.n_rows; row += total_threads_x){
    for(int col = col_init; col < src.n_cols; col += total_threads_y){
      for(int i=0; i<repeat; i++){
        int dst_col = col + (i) * src.n_cols;
        if (dst_col < dst.n_cols) {  // Ensure we do not write out of bounds
          dst(row, dst_col) = src(row, col);  // Copy src element to the repeated positions in dst
        }
      }
    }
  }
}

/**
 * A CUDA kernel function that computes the sum of a matrix along a specified
 * axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param alpha The scaling factor for the sum.
 * @param axis The axis along which to compute the sum (0 for rows, 1 for
 * columns).
 */
__global__ void MatSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha,
                       int axis)
{
  // TODO: implement this kernel function
  if(axis == 1){
    // sum along rows -> (n_rows, 1)
    int total_threads = gridDim.x * blockDim.x;
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int row = start_idx; row < src.n_rows; row += total_threads){
      nn_real sum = 0;
      for(int col=0; col<src.n_cols; col++){
        sum += src(row, col);
      }
      dst(row, 0) = alpha * sum;
    }
  }
  else if (axis == 0){
    // sum along columns -> (1, n_cols)
    int total_threads = gridDim.y * blockDim.y;
    int start_idx = blockIdx.y * blockDim.y + threadIdx.y;
    for(int col = start_idx; col < src.n_cols; col += total_threads){
      nn_real sum = 0;
      for(int row=0; row<src.n_rows; row++){
        sum += src(row, col);
      }
      dst(0, col) = alpha * sum;
    }
  }

}

/**
 * A CUDA kernel function that applies the softmax function along a specified
 * axis to a matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param axis The axis along which to apply the softmax function (0 for rows, 1
 * for columns).
 */
__global__ void MatSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis)
{
  /**
   * TODO: implement this kernel function
   * Hint: Use Exp() from common.h
   * A possible implementation is to have one thread per row (or  column,
   * depending on axis), compute the sum of exponentials of all elements in
   * the row by iterating through elements in the row, and then replace
   * dst(row, col) with the exponential of src(row, col) divided by the sum.
   */
    if(axis == 1){
    // col
    // int thread_Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < src.n_rows; idx += total_threads){
      nn_real sum = 0;
      if(idx < src.n_rows){
        for(int i=0; i<src.n_cols; i++){
          sum += Exp(src(idx, i));
        }

        for(int j=0; j<src.n_cols; j++){
          dst(idx, j) = Exp(src(idx, j))/sum;
        }
      }
    }    
  }
  else if(axis == 0){
    // row
    // int thread_Idx = blockIdx.y * blockDim.y + threadIdx.y;
    int total_threads = gridDim.y * blockDim.y;
    for(int idx = blockIdx.y * blockDim.y + threadIdx.y; idx < src.n_cols; idx += total_threads){
      nn_real sum = 0;
      if (idx < src.n_cols){
        for(int i=0; i<src.n_rows; i++){
          sum += Exp(src(i, idx));
        }
        for(int j=0; j<src.n_rows; j++){
          dst(j, idx) = Exp(src(j, idx))/sum;
        }
      }
    }
  }
}

/**
 * A CUDA kernel function that computes the cross-entropy loss between predicted
 * and true labels on the GPU.
 *
 * @param y_pred The predicted label matrix.
 * @param y The true label matrix.
 * @param loss The output loss matrix.
 */
__global__ void MatCrossEntropyLoss(DeviceMatrix y_pred, DeviceMatrix y,
                                    DeviceMatrix loss)
{
  /**
   * TODO: implement this kernel function
   * Hint: This kernel computes loss = -y * log(y_pred) where * denotes
   * element-wise multiplication and log is applied element-wise. Use
   * Log() from common.h
   */
  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < y.n_rows; row += total_threads_x){
    for(int col = blockIdx.y * blockDim.y + threadIdx.y; col < y.n_cols; col += total_threads_y){
        loss(row, col) = -y(row, col) * Log(y_pred(row, col));
    }
  }

}

/**
 * A CUDA kernel function that performs element-wise arithmetic operations on
 * two matrices on the GPU. A = alpha * (A + beta * B)
 *
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param alpha The scaling factor for the first input matrix.
 * @param beta The scaling factor for the second input matrix.
 */
__global__ void MatElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha,
                             nn_real beta)
{
  // TODO: implement this kernel function

  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < A.n_rows; row += total_threads_x){
    for(int col = blockIdx.y * blockDim.y + threadIdx.y; col < A.n_cols; col += total_threads_y){
        A(row, col) = alpha * (A(row, col) + beta * B(row, col));
    }
  }

}

/**
 * A CUDA kernel function that computes the element-wise square of a matrix on
 * the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSquare(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this kernel function
  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < src.n_rows; row += total_threads_x){
    for(int col = blockIdx.y * blockDim.y + threadIdx.y; col < src.n_cols; col += total_threads_y){
      if(row < src.n_rows && col < src.n_cols){
        dst(row, col) = src(row, col) * src(row, col);
      }
    }
  }
}

/**
 * A CUDA kernel function that computes backpropagation for sigmoid function on
 * the GPU.
 *
 * @param da1 The upstream derivative matrix.
 * @param a1 The activation matrix.
 * @param dz1 The output derivative matrix.
 */
__global__ void MatSigmoidBackProp(DeviceMatrix da1, DeviceMatrix a1,
                                   DeviceMatrix dz1)
{
  /**
   * TODO: implement this kernel function
   * Hint: This kernel computes dz1 = da1 * a1 * (1 - a1), where * denotes
   * element-wise multiplication.
   */

  int total_threads_x = gridDim.x * blockDim.x;
  int total_threads_y = gridDim.y * blockDim.y;
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < a1.n_rows; row += total_threads_x){
    for(int col = blockIdx.y * blockDim.y + threadIdx.y; col < a1.n_cols; col += total_threads_y){
      if(row < a1.n_rows && col < a1.n_cols){
        dz1(row, col) = da1(row, col) * a1(row, col) * (1 - a1(row, col));
      }
    }
  }
}

__global__ void Warmup() {}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                       GPU kernel wrappers                        //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

void DSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this function

  // ensure the src and dst matrices have the same dimensions
  assert (src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);

  // launch kernel 
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (src.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (src.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatSigmoid<<<gridSize, blockSize>>>(src, dst);

  // Check for any errors launching the kernel
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
  }

  cudaDeviceSynchronize();
  CHECK_LAUNCH("DSigmoid");
}

void DRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat)
{
  // TODO: implement this function
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (src.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (src.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatRepeatColVec<<<gridSize, blockSize>>>(src, dst, repeat);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DRepeatColVec");
}

void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis)
{
  // TODO: implement this function
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (src.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (src.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatSum<<<gridSize, blockSize>>>(src, dst, alpha, axis);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DSum");
}

void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis)
{
  // TODO: implement this function
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (src.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (src.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatSoftmax<<<gridSize, blockSize>>>(src, dst, axis);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DSoftmax");
}

void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss)
{
  /**
   * TODO: implement this function
   * Hint: Initialize a temporary matrix T to store the loss and then call
   * MatCrossEntropyLoss. Call DSum twice to compute the sum of all elements
   * in T.
   */

  DeviceMatrix T(y.n_rows, y.n_cols); //to store the loss
  DeviceMatrix T2(y.n_rows, y.n_cols); //to store the loss

  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (y.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (y.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatCrossEntropyLoss<<<gridSize, blockSize>>>(y_pred, y, T);
  DSum(T, T2, 1, 0);
  DSum(T2, loss, 1, 1);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DCELoss");
}

void DElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha, nn_real beta)
{
  // TODO: implement this function
  assert (A.n_rows == B.n_rows && A.n_cols == B.n_cols);
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (A.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (A.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatElemArith<<<gridSize, blockSize>>>(A, B, alpha, beta);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DElemArith");
}

void DSquare(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this function
  assert (src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (src.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (src.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatSquare<<<gridSize, blockSize>>>(src, dst);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DSquare");
}

void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1)
{
  // TODO: implement this function
  dim3 blockSize(32, 32);
  int blocks_per_grid_row = (da1.n_rows + blockSize.x - 1) / blockSize.x;
  int blocks_per_grid_col = (da1.n_cols + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(blocks_per_grid_row, blocks_per_grid_col);

  MatSigmoidBackProp<<<gridSize, blockSize>>>(da1, a1, dz1);
  cudaDeviceSynchronize();
  CHECK_LAUNCH("DSigmoidBackprop");
}

void DWarmup() { Warmup<<<1, 1>>>(); }
