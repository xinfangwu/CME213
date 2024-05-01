#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // V TODO
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int border = order/2;
    if(thread_idx < nx * ny){
        int x = border + (thread_idx % nx);
        int y = border + (thread_idx / nx);
        int out = gx * y + x;

        next[out] = Stencil<order>(&curr[out], gx, xcfl, ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // V TODO: Declare variables/Compute parameters.
    int order = params.order();
    int nx_ = params.nx();
    int ny_ = params.ny();
    int gx_ = params.gx();
    int gy_ = params.gy();
    double xcfl_ = params.xcfl(); 
    double ycfl_ = params.ycfl(); 
    int blocksize = 1024;
    int gridsize = (nx_*ny_ + blocksize - 1)/blocksize;

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // V TODO: Apply stencil.
        if(order == 2){
            gpuStencilGlobal<2><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        else if(order == 4){
            gpuStencilGlobal<4><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        else if(order == 8){
            gpuStencilGlobal<8><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    size_t thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x; 
    size_t thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int border = order/2;
    int init_y = thread_idx_y * numYPerStep;
    int end_y = (thread_idx_y + 1) * numYPerStep;
    
    if(thread_idx_x < nx){
        int x = border + thread_idx_x;

        for(int y = init_y; y<end_y; y++){
            if(y < ny){
                int out = gx * (y+border) + x;
                next[out] = Stencil<order>(&curr[out], gx, xcfl, ycfl);
            }
        }

    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int order = params.order();
    int nx_ = params.nx();
    int ny_ = params.ny();
    int gx_ = params.gx();
    int gy_ = params.gy();
    double xcfl_ = params.xcfl(); 
    double ycfl_ = params.ycfl(); 
    
    // TODO: use a 2D grid with 𝑛𝑥 × 𝑛𝑦/numYPerStep threads total
    // max threads in block = 1024 
    #define numYPerStep 16
    int numThread_x = 128;
    int numThread_y = 8;

    int numBlock_x = (nx_ + numThread_x -1)/ numThread_x;
    int numBlock_y = (ny_ + numThread_y -1)/ numThread_y;
    
    dim3 blocksize(numThread_x, numThread_y);
    dim3 gridsize(numBlock_x, numBlock_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        if(order == 2){
            gpuStencilBlock<2, numYPerStep><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        else if(order == 4){
            gpuStencilBlock<4, numYPerStep><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        else if(order == 8){
            gpuStencilBlock<8, numYPerStep><<<gridsize, blocksize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_);
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    __shared__ float shared_data[side][side];
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_idx_x < gx && thread_idx_y < gy){
        shared_data[threadIdx.x][threadIdx.y] = curr[gy * thread_idx_y + thread_idx_x];
    }

    __syncthreads();

    if (thread_idx_x < gx && thread_idx_y < gy){
        next[gy * thread_idx_y + thread_idx_x] =  shared_data[threadIdx.x][threadIdx.y]
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    int order_ = params.order();
    int nx_ = params.nx();
    int ny_ = params.ny();
    int gx_ = params.gx();
    int gy_ = params.gy();
    double xcfl_ = params.xcfl(); 
    double ycfl_ = params.ycfl(); 

    // TODO: Declare variables/Compute parameters.
    int numThread_x = 64; // why recommend 64?
    int numThread_y = 8;

    int numBlock_x = (nx_ + numThread_x -1)/ numThread_x;
    int numBlock_y = (ny_ + numThread_y -1)/ numThread_y;

    dim3 blocksize(numThread_x, numThread_y);
    dim3 gridsize(numBlock_x, numBlock_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        gpuStencilShared<side, order_><<<blocksize, gridsize>>>(next_grid.dGrid_, curr_grid.dGrid_, gx_, nx_, ny_, xcfl_, ycfl_)
        cudaDeviceSynchronize();
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}

