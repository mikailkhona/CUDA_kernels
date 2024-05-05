#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel for the Up-Sweep (Reduction) Phase
__global__ void upsweep(int *data, int n, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (idx + 1) * stride * 2 - 1;
    if (index < n) {
        data[index] += data[index - stride];
    }
}

// Kernel for the Down-Sweep Phase
__global__ void downsweep(int *data, int n, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (idx + 1) * stride * 2 - 1;
    if (index < n) {
        int temp = data[index - stride];
        data[index - stride] = temp;
        data[index] += temp;
    }
}

// Function to execute parallel scan using the Blelloch algorithm
void parallel_scan(int *data, int n) {
    int *d_data;
    size_t size = n * sizeof(int);
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Up-Sweep Phase
    for (int stride = 1; stride < n; stride <<= 1) {
        upsweep<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, n, stride);
        cudaDeviceSynchronize();
    }

    // Set the last element to zero before Down-Sweep
    cudaMemset(&d_data[n - 1], 0, sizeof(int));

    // Down-Sweep Phase
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        downsweep<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, n, stride);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main() {
    const int n = 16;
    int data[n] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 4, 5, 1, 0, 2, 6, 8};

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;

    parallel_scan(data, n);

    std::cout << "Prefix sum (Exclusive scan): ";
    for (int i = 0; i < n; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;

    return 0;
}
