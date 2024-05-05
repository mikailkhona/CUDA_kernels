#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel to perform Bitonic Sort step
__global__ void bitonic_sort_step(int *arr, int size, int step, int sub_step) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pair_distance = 1 << (sub_step - 1);
    bool increasing_order = ((tid & step) == 0);

    unsigned int index1 = tid * 2 * pair_distance;
    unsigned int index2 = index1 + pair_distance;

    if (index1 < size && index2 < size) {
        bool condition = (arr[index1] > arr[index2]) == increasing_order;
        if (condition) {
            int temp = arr[index1];
            arr[index1] = arr[index2];
            arr[index2] = temp;
        }
    }
}

// Function to execute Bitonic Sort on GPU
void bitonic_sort(int *arr, int size) {
    int *d_arr;
    cudaMalloc((void **)&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int step = 2; step <= size; step <<= 1) {
        for (int sub_step = step >> 1; sub_step > 0; sub_step >>= 1) {
            bitonic_sort_step<<<num_blocks, THREADS_PER_BLOCK>>>(d_arr, size, step, sub_step);
        }
    }

    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    const int size = 16;
    int arr[size] = {3, 7, 4, 8, 6, 2, 1, 5, 10, 9, 12, 14, 11, 15, 13, 16};

    std::cout << "Original array: ";
    for (int i = 0; i < size; ++i) std::cout << arr[i] << " ";
    std::cout << std::endl;

    bitonic_sort(arr, size);

    std::cout << "Sorted array: ";
    for (int i = 0; i < size; ++i) std::cout << arr[i] << " ";
    std::cout << std::endl;

    return 0;
}
