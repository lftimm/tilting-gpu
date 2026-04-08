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

#define HIP_IDX(cord) (blockDim.cord * blockIdx.cord + threadIdx.cord)

__global__ void matrixAdd(float* a, float* b, float* c, int rows, int cols) {
    int xId = HIP_IDX(x);
    int yId = HIP_IDX(y);

    if (xId >= cols || yId >= rows) return;

    int linearIdx = yId * cols + xId;

    c[linearIdx] = a[linearIdx] + b[linearIdx];
}

int main() {

    rocrand_generator generator;
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_set_stream(generator, hipStreamDefault);
    
    int r = 16;
    int c = 16;
    int vecSize = r*c;

    float *d_a,*d_b,*d_c;

    HIP_CHECK(hipMalloc(&d_a,sizeof(float)*vecSize));
    HIP_CHECK(hipMalloc(&d_b,sizeof(float)*vecSize));
    HIP_CHECK(hipMalloc(&d_c,sizeof(float)*vecSize));

    dim3 blockSize{16,16};
    dim3 gridSize{(c+blockSize.x-1),(r + blockSize.y-1)};

    rocrand_generate_normal(generator,d_a,vecSize,0.f,1.f);
    rocrand_generate_normal(generator,d_b,vecSize,0.f,1.f);

    matrixAdd<<<gridSize,blockSize>>>(d_a, d_b, d_c, r, c);

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    std::cout << "Matrix addition completed successfully!" << std::endl;

    return 0;
}
