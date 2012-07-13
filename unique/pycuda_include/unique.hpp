#include <stdio.h>
#include "bitonic_sort.hpp"

namespace unique {
    template <class T>
    __device__ int unique(int size, T *data) {
        __shared__ int unique_index;
        if(threadIdx.x == 0) unique_index = 0;

        bitonic_sort::bitonic_sort<T>(size, &data[0], true);

        T value = 0;
        T other_value = 0;

        int passes = ceil((float)(size) / blockDim.x);
        for(int k = 0; k < passes; k++) {
            int i = k * blockDim.x + threadIdx.x;
            if(i < size) {
                value = data[i];
                other_value = value;
                if(i > 0 && data[i] != data[i - 1]) {
                    /* If we're not the first thread, save current element and
                    * the element with the next lowest ID.
                    */
                    other_value = data[i - 1];
                }
            }
            __syncthreads();

            if(i == 0 || (i < size && value != other_value)) {
                /* If value does not match other_value, it follows that
                * the element at index i is the last position holding
                * 'value'.  Thus, we store the value in the next available
                * position.
                */
                int index = atomicAdd(&unique_index, 1);
                data[index] = value;
            }
            __syncthreads();
        }

        return unique_index;
    }
}
