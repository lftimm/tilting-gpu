#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <stdexcept>

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

__global__ void vectorAdd(float* a, float* b, float* c, int N) {

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx>=N) 
        return;

    c[idx] = a[idx] + b[idx];
}

int main(int argc, char** argv) {

    if(argc != 2) 
        throw std::runtime_error("usage: ./VectorAdd {vector size}");
    
    const int vecSize = std::atoi(argv[1]);
    const size_t vecMemSize = sizeof(float)*vecSize;

    // Setup generator
    rocrand_generator generator;
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_set_stream(generator, hipStreamDefault);
    rocrand_initialize_generator(generator);

    // Allocate memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a,vecMemSize));
    HIP_CHECK(hipMalloc(&d_b,vecMemSize));
    HIP_CHECK(hipMalloc(&d_c,vecMemSize));

    // Generate values
    rocrand_generate_normal(generator,d_a,vecSize,0.f,1.f);
    rocrand_generate_normal(generator,d_b,vecSize,0.f,1.f);

    // Launch kernal
    dim3 threadsPerBlock{256};
    dim3 numBlocks(vecSize/threadsPerBlock.x + 1);

    auto start = std::chrono::high_resolution_clock::now();

    vectorAdd<<<numBlocks,threadsPerBlock>>>(d_a,d_b,d_c,vecSize);

    HIP_CHECK(hipDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "took: " << duration.count() << "ms to sum." << std::endl;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    rocrand_destroy_generator(generator);

    return 0;
}
