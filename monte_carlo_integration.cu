#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

const int NUM_SAMPLES = 1000000;
const int THREADS_PER_BLOCK = 256;

// CUDA kernel to initialize random states
__global__ void init_random_states(curandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel to compute Monte Carlo integration
__global__ void monte_carlo_integration(float *results, curandState *states, int num_samples) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = states[idx];

    float sum = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        float x = curand_uniform(&local_state);
        sum += x * x; // Function: x^2 in the range [0, 1]
    }

    results[idx] = sum / num_samples;
    states[idx] = local_state;
}

// Host function to run the Monte Carlo integration on GPU
float monte_carlo_integrate(int num_samples) {
    int num_threads = THREADS_PER_BLOCK;
    int num_blocks = (num_samples + num_threads - 1) / num_threads;

    curandState *d_states;
    float *d_results;
    float *h_results = new float[num_blocks];

    // Allocate memory
    cudaMalloc(&d_states, num_blocks * num_threads * sizeof(curandState));
    cudaMalloc(&d_results, num_blocks * num_threads * sizeof(float));

    // Initialize random states
    init_random_states<<<num_blocks, num_threads>>>(d_states, time(NULL));

    // Run Monte Carlo integration kernel
    monte_carlo_integration<<<num_blocks, num_threads>>>(d_results, d_states, num_samples / (num_blocks * num_threads));
    
    // Copy results back to host
    cudaMemcpy(h_results, d_results, num_blocks * num_threads * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up all the partial results
    float sum = 0.0f;
    for (int i = 0; i < num_blocks * num_threads; ++i) {
        sum += h_results[i];
    }

    // Cleanup
    delete[] h_results;
    cudaFree(d_states);
    cudaFree(d_results);

    // Calculate the final integral value (area)
    return sum / num_blocks;
}

int main() {
    int num_samples = NUM_SAMPLES;
    float result = monte_carlo_integrate(num_samples);

    std::cout << "Estimated integral value: " << result << std::endl;

    return 0;
}
