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
template <int order>
__device__ float Stencil(const float *curr, int width, float xcfl, float ycfl)
{
    switch (order)
    {
    case 2:
        return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
               ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

    case 4:
        return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0] + 16.f * curr[-1] - curr[-2]) + ycfl * (-curr[2 * width] + 16.f * curr[width] - 30.f * curr[0] + 16.f * curr[-width] - curr[-2 * width]);

    case 8:
        return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3] - 1008.f * curr[2] + 8064.f * curr[1] - 14350.f * curr[0] + 8064.f * curr[-1] - 1008.f * curr[-2] + 128.f * curr[-3] - 9.f * curr[-4]) + ycfl * (-9.f * curr[4 * width] + 128.f * curr[3 * width] - 1008.f * curr[2 * width] + 8064.f * curr[width] - 14350.f * curr[0] + 8064.f * curr[-width] - 1008.f * curr[-2 * width] + 128.f * curr[-3 * width] - 9.f * curr[-4 * width]);

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
template <int order>
__global__ void gpuStencilGlobal(float *next, const float *__restrict__ curr, int gx, int nx, int ny,
                                 float xcfl, float ycfl)
{
    const int borderSize = order / 2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nx * ny)
        return;

    const int x = tid % nx + borderSize;
    const int y = tid / nx + borderSize;

    const int gl = y * gx + x;
    next[gl] = Stencil<order>(&curr[gl], gx, xcfl, ycfl);
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
double gpuComputationGlobal(Grid &curr_grid, const simParams &params)
{

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    const int gx = params.gx();
    const int nx = params.nx();
    const int ny = params.ny();

    const int N = nx * ny;

    const int num_threads = 1024;
    const int num_blocks = (N + num_threads - 1) / num_threads;

    event_pair timer;
    start_timer(&timer);

    for (int i = 0; i < params.iters(); ++i)
    {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // apply stencil
        switch (params.order())
        {
        case 2:
            gpuStencilGlobal<2><<<num_blocks, num_threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;

        case 4:
            gpuStencilGlobal<4><<<num_blocks, num_threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;

        case 8:
            gpuStencilGlobal<8><<<num_blocks, num_threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;
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
template <int order, int numYPerStep>
__global__ void gpuStencilBlock(float *next, const float *__restrict__ curr, int gx, int nx, int ny,
                                float xcfl, float ycfl)
{
    const unsigned borderSize = order >> 1;
    const unsigned tidx = blockIdx.x * blockDim.x + threadIdx.x + borderSize;

    if (tidx >= nx + borderSize)
    {
        return;
    }

    const unsigned tidy = blockIdx.y * blockDim.y * numYPerStep + threadIdx.y +
                          borderSize;
    const size_t g_start = gx * tidy + tidx;
    const unsigned g_stride = gx * blockDim.y;

    const unsigned g_end1 = tidy + blockDim.y * numYPerStep;
    // end of y loop if we compute numYPerStep iterations
    const unsigned g_end2 = ny + borderSize;
    // end of grid along y dimension

    if (g_end1 <= g_end2)
    {
        // We can safely perform numYPerStep iterations
        unsigned gl = g_start;

        for (unsigned i = 0; i < numYPerStep; ++i)
        {
            next[gl] = Stencil<order>(&curr[gl], gx, xcfl, ycfl);
            gl += g_stride;
        }
    }
    else
    {
        const size_t g_end = gx * (ny + borderSize) + tidx;
        // The thread stops once it reaches the end of the grid

        for (unsigned gl = g_start; gl < g_end; gl += g_stride)
        {
            next[gl] = Stencil<order>(&curr[gl], gx, xcfl, ycfl);
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
double gpuComputationBlock(Grid &curr_grid, const simParams &params)
{

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    const int numThreadx = 64; /* opt 1: 64; opt 2: 32 */
    const int numThready = 8;
    const int numYPerStep = 8; /* opt 1: 8; opt 2: 4 */

    const int gx = params.gx();
    const int nx = params.nx();
    const int ny = params.ny();

    dim3 threads(numThreadx, numThready);
    dim3 blocks((nx + numThreadx - 1) / numThreadx,
                (ny + numThready * numYPerStep - 1) /
                    (numYPerStep * numThready));

    event_pair timer;
    start_timer(&timer);

    for (int i = 0; i < params.iters(); ++i)
    {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // apply stencil
        switch (params.order())
        {
        case 2:
            gpuStencilBlock<2, numYPerStep><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;

        case 4:
            gpuStencilBlock<4, numYPerStep><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;

        case 8:
            gpuStencilBlock<8, numYPerStep><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny,
                params.xcfl(), params.ycfl());
            break;
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
template <int side, int order>
__global__ void gpuStencilShared(float *next, const float *__restrict__ curr, int gx, int gy,
                                 float xcfl, float ycfl)
{
    const int borderSize = order / 2;
    const int usefulSide = side - order;
    const int xpos = threadIdx.x;
    const int ylane = threadIdx.y; // See the definition of ypos below
    const int globalXPos = blockIdx.x * usefulSide + xpos;
    const int numYPerStep = side / blockDim.y;
    const int localStride = blockDim.y * side;
    const int globalStride = blockDim.y * gx;

    __shared__ float smem[side * side];

    // Use threads to load our slice into smem
    if (globalXPos < gx)
    {
        int localOffset = ylane * side + xpos;
        int globalOffset = (blockIdx.y * usefulSide + ylane) * gx +
                           globalXPos;

        if (side <= gy - blockIdx.y * usefulSide)
        {
            // This block of threads will perform a full
            // side x side update with numYPerStep iterations
            for (int i = 0; i < numYPerStep; ++i)
            {
                smem[localOffset] = curr[globalOffset];
                localOffset += localStride;
                globalOffset += globalStride;
            }
        }
        else
        {
            const int ypos_end = gy - blockIdx.y * usefulSide;
            // End of y iterations for this block
            const int local_end = ypos_end * side + xpos;
            // End of loop using localOffset

            for (; localOffset < local_end; localOffset += localStride)
            {
                smem[localOffset] = curr[globalOffset];
                globalOffset += globalStride;
            }
        }

        // Code variant which does not use a separate loop with numYPerStep
        // performance is lower

        // int localOffset = ylane * side + xpos;
        // int globalOffset = (blockIdx.y * usefulSide + ylane) * gx +
        //                    globalXPos;

        // const int ypos_end = min(side, gy - blockIdx.y * usefulSide);
        // // End of y iterations for this block
        // const int local_end = ypos_end * side + xpos;
        // // End of loop using localOffset

        // for(; localOffset < local_end; localOffset += localStride) {
        //     smem[localOffset] = curr[globalOffset];
        //     globalOffset += globalStride;
        // }
    }

    __syncthreads();

    // Now that everything is loaded in smem, do the stencil calculation.
    if (globalXPos < gx - borderSize && xpos >= borderSize &&
        xpos < side - borderSize)
    {

        int localOffset = (ylane + borderSize) * side + xpos;
        int globalOffset = (blockIdx.y * usefulSide + ylane + borderSize) * gx + globalXPos;

        const int ypos_end = min(side - borderSize,
                                 gy - borderSize - blockIdx.y * usefulSide);
        const int local_end = ypos_end * side + xpos;

        for (; localOffset < local_end; localOffset += localStride)
        {
            next[globalOffset] =
                Stencil<order>(&smem[localOffset], side, xcfl, ycfl);
            globalOffset += globalStride;
        }

        // Code variant with a loop over a separate index i
        // performand is lower

        // const int i_end = (local_end - localOffset + localStride-1) / localStride;
        // for(int i = 0; i<i_end; ++i) {
        //     next[globalOffset] =
        //         Stencil<order>(&smem[localOffset], side, xcfl, ycfl);
        //     localOffset += localStride;
        //     globalOffset += globalStride;
        // }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template <int order>
double gpuComputationShared(Grid &curr_grid, const simParams &params)
{

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    const int numThreadx = 64; // 64;
    const int numThready = 2;  // 8;

    const int gx = params.gx();
    const int gy = params.gy();
    const int nx = params.nx();
    const int ny = params.ny();

    const int smemSide = numThreadx;
    const int usefulsmemSide = smemSide - order;

    const int numBlocksX = (nx + usefulsmemSide - 1) / usefulsmemSide;
    const int numBlocksY = (ny + usefulsmemSide - 1) / usefulsmemSide;

    dim3 threads(numThreadx, numThready);
    dim3 blocks(numBlocksX, numBlocksY);

    event_pair timer;
    start_timer(&timer);

    for (int i = 0; i < params.iters(); ++i)
    {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // apply stencil
        gpuStencilShared<smemSide, order><<<blocks, threads>>>(
            next_grid.dGrid_, curr_grid.dGrid_, gx, gy,
            params.xcfl(), params.ycfl());

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}
