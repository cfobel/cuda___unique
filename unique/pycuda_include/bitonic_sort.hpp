#ifndef ___BITONIC_SORT__HPP___
#define ___BITONIC_SORT__HPP___

namespace bitonic_sort {
    template <class T>
    __device__ inline void Comparator(
        volatile T& keyA,
        volatile T& keyB,
        bool direction
    ){
        T t;
        if((keyA > keyB) == direction){
            t = keyA;
            keyA = keyB;
            keyB = t;
        }
    }


    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>

    template <class T>
    __device__ void dump_data(int size, volatile T *data) {
        syncthreads();
        if(threadIdx.x == 0) {
            printf("[");
            for(int i = 0; i < size; i++) {
                printf("%d, ", data[i]);
            }
            printf("]\n");
        }
    }


    template <class T>
    __device__ T greatest_power_of_two_less_than(T n) {
        T k = 1;
        while(k < n) {
            k = k << 1;
        }
        return k >> 1;
    }


    template <class T>
    __device__ void do_bitonic_sort(int size, volatile T *data, bool direction) {
        int passes = ceil((float)size / 2 / blockDim.x);
        for(uint slice_size = 2; slice_size < size; slice_size <<= 1){
            for(uint stride = slice_size / 2; stride > 0; stride >>= 1){
            //for(uint stride = greatest_power_of_two_less_than(slice_size);
            //        stride > 0; stride >>= 1) {
                __syncthreads();
                for(int k = 0; k < passes; k++) {
                    int i = k * blockDim.x + threadIdx.x;

                    //Bitonic merge
                    uint ddd = direction ^ ((i & (slice_size / 2)) != 0 );

                    /*
                    * The following line is equivalent to:
                    *     uint pos = 2 * threadIdx.x - (threadIdx.x % stride);
                    */
                    uint pos = 2 * i - (i & (stride - 1));
                    if(pos + stride < size) {
                        Comparator<T>(data[pos + 0], data[pos + stride], ddd);
                    }
                }
            }
        }
    }


    template <class T>
    __device__ void do_bitonic_merge(int size, volatile T *data, bool direction) {
        //ddd == direction for the last bitonic merge step
        {
            int passes = ceil((float)size / 2 / blockDim.x);
            for(uint stride = size / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                for(int k = 0; k < passes; k++) {
                    int i = k * blockDim.x + threadIdx.x;

                    uint pos = 2 * i - (i & (stride - 1));
                    if(pos + stride < size) {
                        Comparator<T>(data[pos + 0], data[pos + stride], direction);
                    }
                }
            }
        }

        __syncthreads();
    }


    template <class T>
    __device__ void bitonic_sort(int size, volatile T *data, bool direction) {
        int two_power_size = 1 << (int)log2((float)size);
        bool ddd = direction;

        /* Sort the first largest possible set of elements from the data
         * whose size is a power of two using bitonic sort.
         */
        do_bitonic_sort(two_power_size, data, ddd);
        do_bitonic_merge(two_power_size, data, ddd);

        int processed = two_power_size;

        if(size > two_power_size) {
            // Perform a simple insertion sort to insert remaining elements
            for(int remaining_index = 0; remaining_index < size - two_power_size;
                    remaining_index++) {
                __syncthreads();
                T compare_value = data[processed];
                __syncthreads();
                T temp;
                int passes = ceil((float)(processed + 1) / blockDim.x);
                for(int k = 0; k < passes; k++) {
                    if(direction) {
                        int i = processed - (k * blockDim.x + threadIdx.x);
                        if(i > 0) {
                            temp = data[i - 1];
                        }
                        __syncthreads();
                        if(i >= 0) {
                            if(i > 0 && temp >= compare_value) {
                                data[i] = temp;
                            } else if(data[i] >= compare_value) {
                                data[i] = compare_value;
                            }
                        }
                    } else {
                        int i = processed - (k * blockDim.x + threadIdx.x);
                        if(i > 0) {
                            temp = data[i - 1];
                        }
                        __syncthreads();
                        if(i >= 0) {
                            if(i > 0 && temp < compare_value) {
                                data[i] = temp;
                            } else if(data[i] <= compare_value) {
                                data[i] = compare_value;
                            }
                        }
                    }
                }
                processed += 1;
            }
        }
    }
}

#endif
