#ifndef _RECURRENCE_CUH
#define _RECURRENCE_CUH

#include "util.cuh"

/**
 * Repeating from the tutorial, just in case you haven't looked at it.
 * "kernels" or __global__ functions are the entry points to code that executes
 * on the GPU. The keyword __global__ indicates to the compiler that this
 * function is a GPU entry point.
 * __global__ functions must return void, and may only be called or "launched"
 * from code that executes on the CPU.
 */

typedef float elem_type;

/**
 * TODO: implement the kernel recurrence.
 * The CPU implementation is in host_recurrence() in main_q1.cu.
 */
__global__ void recurrence(const elem_type* input_array,
                           elem_type* output_array, size_t num_iter,
                           size_t array_length) {
  size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_thread = gridDim.x * blockDim.x;

  for(size_t i = thread_idx; i < array_length; i+= total_thread){
    elem_type z = 0;
    elem_type c = input_array[i];
    for(size_t it=0; it<num_iter; it++){
      z = z * z + c;
    }
    output_array[i] = z;
  }

}

/**
* This function calls the recurrence kernel 
*
* d_input:
*   array with initialized input values
* d_output:
*   array to be filled with output of reccurence code
* num_iter:
*   Number of times we will iterate before checking for convergence
* array_length:
*   Size of arrays d_input and d_output
* block_size:
*   Number of threads in CUDA block
* grid_size:
*   Number of blocks in CUDA call
*/
double doGPURecurrence(const elem_type* d_input, elem_type* d_output,
                       size_t num_iter, size_t array_length, size_t block_size,
                       size_t grid_size) {
  event_pair timer;
  start_timer(&timer);
  // V TODO: launch kernel
  recurrence<<<grid_size, block_size>>>(d_input, d_output, num_iter, array_length);

  check_launch("gpu recurrence");
  return stop_timer(&timer);
}

#endif
