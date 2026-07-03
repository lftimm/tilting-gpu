#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>
#include <print>

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

int main() {

    int devCount;
    HIP_CHECK(hipGetDeviceCount(&devCount));

    std::printf("devCount: %d\n",devCount);
    

    hipDeviceProp_t devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        hipGetDeviceProperties(&devProp, i);

        std::printf("name: %s\n",devProp.name);
        std::printf("maxThreadsPerBlock: %d\n",devProp.maxThreadsPerBlock);
        std::printf("multiProcessorCount: %d\n",devProp.multiProcessorCount);
        std::printf("clockRate: %d\n",devProp.clockRate);
        std::printf("maxThreadsDim[0] (x): %d\n",devProp.maxThreadsDim[0]);
        std::printf("maxThreadsDim[1] (y): %d\n",devProp.maxThreadsDim[1]);
        std::printf("maxThreadsDim[2] (z): %d\n",devProp.maxThreadsDim[2]);
        std::printf("maxGridSize[0] (x): %d\n",devProp.maxGridSize[0]);
        std::printf("maxGridSize[1] (y): %d\n",devProp.maxGridSize[1]);
        std::printf("maxGridSize[2] (z): %d\n",devProp.maxGridSize[2]);
        std::printf("regsPerBlock: %d\n",devProp.regsPerBlock);
        std::printf("warpSize: %d\n",devProp.warpSize);

    }

    return 0;
}
