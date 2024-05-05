#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <cuda_runtime.h>

const int N = 512; // Grid size
const float p = 0.5f; // Percolation probability

// CUDA error checking macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

// CUDA kernel to initialize the random states
__global__ void init_random_states(curandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel to generate a percolation grid
__global__ void generate_grid(int *grid, curandState *states, float p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * N) {
        curandState local_state = states[idx];
        float rand_val = curand_uniform(&local_state);
        grid[idx] = (rand_val < p) ? 1 : 0;
        states[idx] = local_state;
    }
}

// CUDA kernel to flood-fill from the top row
__global__ void flood_fill(int *grid, int *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < 0 || x >= N || y < 0 || y >= N) return;

    int idx = x + y * N;
    if (grid[idx] == 0) return;

    // Mark the current cell as visited
    result[idx] = 1;

    // Check neighboring cells
    int neighbors[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < 4; ++i) {
        int nx = x + neighbors[i][0];
        int ny = y + neighbors[i][1];
        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
            int neighbor_idx = nx + ny * N;
            if (grid[neighbor_idx] == 1 && result[neighbor_idx] == 0) {
                result[neighbor_idx] = 1;
                flood_fill<<<1, 1>>>(grid, result);
            }
        }
    }
}

__global__ void check_percolation(int *result, bool *percolated) {
    for (int x = 0; x < N; ++x) {
        if (result[x + (N - 1) * N] == 1) {
            *percolated = true;
        }
    }
}

int main() {
    // Allocate host and device memory
    int *grid, *d_grid;
    int *result, *d_result;
    curandState *d_states;
    bool percolated = false;
    bool *d_percolated;

    grid = new int[N * N];
    result = new int[N * N];
    CUDA_CALL(cudaMalloc((void**)&d_grid, N * N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_result, N * N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_states, N * N * sizeof(curandState)));
    CUDA_CALL(cudaMalloc((void**)&d_percolated, sizeof(bool)));

    // Initialize the random states and generate the grid
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (N + threads_per_block.y - 1) / threads_per_block.y);
    init_random_states<<<blocks_per_grid, threads_per_block>>>(d_states, time(NULL));
    generate_grid<<<blocks_per_grid, threads_per_block>>>(d_grid, d_states, p);
    CUDA_CALL(cudaMemcpy(grid, d_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    // Initialize result grid to zero
    CUDA_CALL(cudaMemset(d_result, 0, N * N * sizeof(int)));

    // Start flood-fill from the top row
    for (int x = 0; x < N; ++x) {
        if (grid[x] == 1) {
            flood_fill<<<1, 1>>>(d_grid, d_result);
        }
    }

    // Check for percolation
    CUDA_CALL(cudaMemset(d_percolated, 0, sizeof(bool)));
    check_percolation<<<1, 1>>>(d_result, d_percolated);
    CUDA_CALL(cudaMemcpy(&percolated, d_percolated, sizeof(bool), cudaMemcpyDeviceToHost));

    std::cout << (percolated ? "Percolates!" : "Does not percolate!") << std::endl;

    // Cleanup
    delete[] grid;
    delete[] result;
    CUDA_CALL(cudaFree(d_grid));
    CUDA_CALL(cudaFree(d_result));
    CUDA_CALL(cudaFree(d_states));
    CUDA_CALL(cudaFree(d_percolated));

    return 0;
}
