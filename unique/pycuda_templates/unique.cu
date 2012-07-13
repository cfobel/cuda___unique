#include <stdio.h>
#include "unique.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void unique_{{ c_type }}(int size, {{ c_type }} *data,
        int *p_unique_index) {
    {{ c_type }} *sh_data = ({{ c_type }} *)&shared_data[0];

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        if(i < size) {
            sh_data[i] = data[i];
        }
    }
    __syncthreads();
    int unique_index = unique::unique(size, sh_data);

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        if(i < size) {
            data[i] = sh_data[i];
        }
    }
    if(threadIdx.x == 0) {
        *p_unique_index = unique_index;
    }
}
{% endfor %}
