#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stack>

const int N = 512; // Grid size
const float p = 0.9f; // Percolation probability

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

// CUDA kernel to initialize the flood-fill result grid
__global__ void initialize_result(int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * N) {
        result[idx] = 0;
    }
}

// Host function to implement iterative flood-fill
void flood_fill(int *grid, int *result) {
    std::stack<int> stack;
    for (int x = 0; x < N; ++x) {
        if (grid[x] == 1) {
            stack.push(x);
            result[x] = 1;
        }
    }

    int neighbors[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    while (!stack.empty()) {
        int idx = stack.top();
        stack.pop();

        int x = idx % N;
        int y = idx / N;

        for (int i = 0; i < 4; ++i) {
            int nx = x + neighbors[i][0];
            int ny = y + neighbors[i][1];
            if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                int neighbor_idx = nx + ny * N;
                if (grid[neighbor_idx] == 1 && result[neighbor_idx] == 0) {
                    stack.push(neighbor_idx);
                    result[neighbor_idx] = 1;
                }
            }
        }
    }
}

// CUDA kernel to copy data to the host flood-fill grid
__global__ void copy_result(int *result, int *d_result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * N) {
        result[idx] = d_result[idx];
    }
}

// CUDA kernel to check for percolation from the top row to the bottom row
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

    // Initialize the result grid to zero
    initialize_result<<<blocks_per_grid, threads_per_block>>>(d_result);

    // Perform flood-fill on the host
    flood_fill(grid, result);
    CUDA_CALL(cudaMemcpy(d_result, result, N * N * sizeof(int), cudaMemcpyHostToDevice));

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

