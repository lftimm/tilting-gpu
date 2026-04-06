#include <hip/hip_runtime.h>
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

struct MatrixView {
    float* data;
    int rows;
    int cols;

    __host__ __device__ MatrixView(float* d, int rows, int cols) : data(d), rows(rows), cols(cols) {};

    __host__ __device__ float& operator()(int j, int k) {
        return data[cols*j+k];
    }

    __host__ __device__ const float& operator()(int j, int k) const {
        return data[cols*j+k];
    }
};

__global__ void mat_ma(MatrixView A)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(y_idx >= A.rows) return;
    if(x_idx >= A.cols) return;

    printf("A(%d,%d) done\n",y_idx,x_idx);
    A(y_idx,x_idx) = 1.f;
}

int main() {

    float* d_a;
    HIP_CHECK(hipMalloc(&d_a,sizeof(float)*9));
    MatrixView d_view{d_a,3,3};

    mat_ma<<<1,dim3(3,3)>>>(d_view);

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_a));

    return 0;
}
