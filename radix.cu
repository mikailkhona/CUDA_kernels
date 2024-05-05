#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void count_kernel(int *d_input, int *d_output, int bit, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_elements) return;

    int mask = 1 << bit;
    d_output[tid] = (d_input[tid] & mask) ? 1 : 0;
}

__global__ void exclusive_scan_kernel(int *d_input, int *d_output, int num_elements) {
    extern __shared__ int temp[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int t = threadIdx.x;

    if (tid < num_elements) {
        temp[2 * t] = d_input[2 * tid];
        temp[2 * t + 1] = d_input[2 * tid + 1];
    } else {
        temp[2 * t] = temp[2 * t + 1] = 0;
    }

    for (int offset = 1; offset < 2 * THREADS_PER_BLOCK; offset *= 2) {
        int value = 0;
        if (t >= offset) value = temp[2 * t - offset];
        __syncthreads();
        temp[2 * t] += value;
        temp[2 * t + 1] += value;
        __syncthreads();
    }

    if (tid < num_elements) {
        d_output[2 * tid] = temp[2 * t];
        d_output[2 * tid + 1] = temp[2 * t + 1];
    }
}

__global__ void scatter_kernel(int *d_input, int *d_output, int *d_scanned, int num_zeros, int bit, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_elements) return;

    int mask = 1 << bit;
    if (d_input[tid] & mask) {
        d_output[num_zeros + d_scanned[tid]] = d_input[tid];
    } else {
        d_output[tid - d_scanned[tid]] = d_input[tid];
    }
}

void radix_sort(int *arr, int num_elements) {
    int *d_input, *d_output, *d_count, *d_scanned;
    int num_bits = 32;

    cudaMalloc(&d_input, num_elements * sizeof(int));
    cudaMalloc(& d_output, num_elements*toad 
