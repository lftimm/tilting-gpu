#include <cmath>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <iostream>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

constexpr float vecSize{1000};
static rocrand_generator generator{};

__global__ void vecMultiply(float *a, float *b, float *r, int n) 
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx >= n) return;
    r[idx] = a[idx] * b[idx];
}

__global__ void reduceSum(float* vec, int n) 
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = 2;
    for(int i = idx; i < stride; stride*=2)
    {
       vec[idx] += vec[idx+i];
    }
}

void fill_device_vector(float** h) 
{
    HIP_CHECK(hipMalloc(h ,sizeof(float)*vecSize));
    rocrand_generate_normal(generator, *h, vecSize, 0, 1.f);
}

int main()
{
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_set_seed(generator, time(0));   

    float* d_a{}; fill_device_vector(&d_a);
    float* d_b{}; fill_device_vector(&d_b);
    float* d_mult{}; HIP_CHECK(hipMalloc(&d_mult,sizeof(float)*vecSize));

    dim3 threadsPerBlock{256};
    dim3 numberOfBlocks(ceil(vecSize/threadsPerBlock.x));

    vecMultiply<<<numberOfBlocks,threadsPerBlock>>>(d_a,d_b,d_mult,vecSize);
    reduceSum<<<numberOfBlocks,threadsPerBlock>>>(d_mult,vecSize);

    HIP_CHECK(hipDeviceSynchronize());

    float h_res{};
    HIP_CHECK(hipMemcpy(&h_res,&d_mult,sizeof(float),hipMemcpyDeviceToHost));

    std::cout << h_res << std::endl;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_mult));
    
    return 0;
}
