#define ARMA_ALLOW_FAKE_GCC
#include <algorithm>
#include <armadillo>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <memory>
#include <cassert>

#include "gpu_func.h"

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n)
{
  assert(n >= 0);
  assert(cpu_data != nullptr);
  nbytes = n * sizeof(nn_real);
  checkCudaErrors(cudaMalloc(&data, nbytes));
  checkCudaErrors(cudaMemcpy(data, cpu_data, nbytes, cudaMemcpyHostToDevice));
}

DeviceAllocator::DeviceAllocator(int n)
{
  assert(n >= 0);
  nbytes = n * sizeof(nn_real);
  checkCudaErrors(cudaMalloc(&data, nbytes));
}

DeviceAllocator::~DeviceAllocator()
{
  if (data != nullptr)
    checkCudaErrors(cudaFree(data));
}

void DeviceAllocator::to_cpu(nn_real *cpu_data)
{
  assert(data != nullptr && cpu_data != nullptr);
  checkCudaErrors(cudaMemcpy(cpu_data, data, nbytes, cudaMemcpyDeviceToHost));
}

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols)
{
  assert(n_rows >= 0 && n_cols >= 0);
  this->allocator = std::make_shared<DeviceAllocator>(n_rows * n_cols);
  this->data = this->allocator->data;
  this->n_rows = n_rows;
  this->n_cols = n_cols;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat)
{
  this->allocator = std::make_shared<DeviceAllocator>(
      cpu_mat.memptr(), cpu_mat.n_rows * cpu_mat.n_cols);
  this->data = this->allocator->data;
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
}

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat)
{
  allocator->to_cpu(cpu_mat.memptr());
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col)
{
  // assert(data != nullptr);
  return data[col * n_rows + row];
}

int DeviceMatrix::total_bytes()
{
  return allocator->nbytes;
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                           CUDA kernels                           //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

// /**
//  * A CUDA kernel function that prints a matrix to the console on the GPU.
//  *
//  * @param mat The input matrix.
//  * @param name The name of the matrix.
//  * @param trans Whether to print the matrix in transposed form.
//  */
// __global__ void PrintMatrix(Matrix mat, char name, bool trans)
// {
//     printf("\nGPU Matrix %c\n", name);
//     for (int i = 0; i < (trans ? mat.n_cols : mat.n_rows); i++)
//     {
//         for (int j = 0; j < (trans ? mat.n_rows : mat.n_cols); j++)
//         {
//             printf("%e ", mat(i, j, trans));
//         }
//         printf("\n");
//     }
// }

// void DevicePrintMatrix(Matrix mat, char name, bool trans)
// {
//     PrintMatrix<<<1, 1>>>(mat, name, trans);
//     CHECK_LAUNCH("DevicePrintMatrix");
// }

/**
 * A CUDA kernel function that applies the sigmoid function element-wise to a
 * matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / src.n_cols;
  int j = idx % src.n_cols;
  if (idx < src.n_rows * src.n_cols)
  {
    dst(i, j) = 1.0f / (1.0f + Exp(-src(i, j)));
  }
}

void DSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);

  int block = 256;
  int grid = (src.n_rows * src.n_cols + block - 1) / block;
  MatSigmoid<<<grid, block>>>(src, dst);
  CHECK_LAUNCH("DSigmoid");
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
  // For column-major matrix, use x-axis for row and y-axis for col
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < src.n_rows)
  {
    nn_real value = src(row, 0);
    for (int i = 0; i < repeat; i++)
    {
      dst(row, i) = value;
    }
  }
}

void DRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat)
{
  assert(src.n_cols == 1 && dst.n_cols == repeat);
  int block = 128;
  int grid = (src.n_rows + block - 1) / block;
  MatRepeatColVec<<<grid, block>>>(src, dst, repeat);
  CHECK_LAUNCH("DRepeatColVec");
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
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ((axis == 0 && idx < dst.n_cols) || (axis == 1 && idx < dst.n_rows))
  {
    nn_real sum = 0;
    int len = axis == 0 ? src.n_rows : src.n_cols;
    for (int i = 0; i < len; i++)
    {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      sum += src(row, col);
    }
    int target_row = axis == 0 ? 0 : idx;
    int target_col = axis == 0 ? idx : 0;
    dst(target_row, target_col) = alpha * sum;
  }
}

void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis)
{
  assert(axis == 0 || axis == 1);
  if (axis == 0)
  {
    assert(src.n_cols == dst.n_cols && dst.n_rows == 1);
  }
  else
  {
    assert(src.n_rows == dst.n_rows && dst.n_cols == 1);
  }

  int block = 256;
  int len = axis == 0 ? src.n_cols : src.n_rows;
  int grid = (len + block - 1) / block;
  MatSum<<<grid, block>>>(src, dst, alpha, axis);
  CHECK_LAUNCH("DSum");
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
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (axis == 0 && idx < src.n_cols || axis == 1 && idx < src.n_rows)
  {
    nn_real exp_sum = 0;
    int len = axis == 0 ? src.n_rows : src.n_cols;
    for (int i = 0; i < len; i++)
    {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      exp_sum += Exp(src(row, col));
    }
    for (int i = 0; i < len; i++)
    {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      dst(row, col) = Exp(src(row, col)) / exp_sum;
    }
  }
}

void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis)
{
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);
  assert(axis == 0 || axis == 1);

  int block = 256;
  int len = axis == 0 ? src.n_cols : src.n_rows;
  int grid = (len + block - 1) / block;
  MatSoftmax<<<grid, block>>>(src, dst, axis);
  CHECK_LAUNCH("DSoftmax");
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
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < loss.n_rows && col < loss.n_cols)
  {
    loss(row, col) = -y(row, col) * Log(y_pred(row, col));
  }
}

void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss)
{
  assert(y.n_rows == y_pred.n_rows && y.n_cols == y_pred.n_cols);

  constexpr int block_size = 32;
  dim3 block(block_size, block_size);
  dim3 grid((y.n_rows + block_size - 1) / block_size,
            (y.n_cols + block_size - 1) / block_size);
  DeviceMatrix loss_mat(y.n_rows, y.n_cols);
  MatCrossEntropyLoss<<<grid, block>>>(y_pred, y, loss_mat);

  DeviceMatrix loss_vec(y.n_rows, 1);
  DSum(loss_mat, loss_vec, static_cast<nn_real>(1), 1);
  DSum(loss_vec, loss, static_cast<nn_real>(1), 0);
  CHECK_LAUNCH("DCELoss");
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
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / A.n_cols;
  int j = idx % A.n_cols;
  if (idx < A.n_rows * A.n_cols)
    A(i, j) = alpha * (A(i, j) + beta * B(i, j));
}

void DElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha, nn_real beta)
{
  assert(A.n_rows == B.n_rows && A.n_cols == B.n_cols);
  int block = 256;
  int grid = (A.n_rows * A.n_cols + block - 1) / block;
  MatElemArith<<<grid, block>>>(A, B, alpha, beta);
  CHECK_LAUNCH("DElemArith");
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / src.n_cols;
  int j = idx % src.n_cols;
  if (idx < src.n_rows * src.n_cols)
  {
    nn_real value = src(i, j);
    dst(i, j) = value * value;
  }
}

void DSquare(DeviceMatrix src, DeviceMatrix dst)
{
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);
  int block = 256;
  int grid = (src.n_rows * src.n_cols + block - 1) / block;
  MatSquare<<<grid, block>>>(src, dst);
  CHECK_LAUNCH("DSquare");
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / da1.n_cols;
  int j = idx % da1.n_cols;
  if (idx < dz1.n_rows * dz1.n_cols)
  {
    dz1(i, j) = da1(i, j) * a1(i, j) * (1 - a1(i, j));
  }
}

void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1)
{
  assert(da1.n_rows == a1.n_rows && da1.n_cols == a1.n_cols);
  assert(da1.n_rows == dz1.n_rows && da1.n_cols == dz1.n_cols);

  int block = 256;
  int grid = (da1.n_rows * da1.n_cols + block - 1) / block;
  MatSigmoidBackProp<<<grid, block>>>(da1, a1, dz1);
  CHECK_LAUNCH("DSigmoidBackprop");
}

__global__ void Warmup() {}

void DWarmup() { Warmup<<<1, 1>>>(); }