//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_KERNELS_H
#define TENSORLIBRARY_KERNELS_H

#include <nvrtc.h>
#include <cuda.h>
#include <iostream>

#define NUM_THREADS 32
#define NUM_BLOCKS 32
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
} while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
} while(0)

extern const char *saxpy;
extern const char *kernel_add;
extern const char *kernel_neg;
extern const char *kernel_mul;
extern const char *kernel_div;
extern const char *kernel_matmul;
extern const char *kernel_relu;
extern const char *kernel_log;
extern const char *kernel_exp;
extern const char *kernel_sum;
extern const char *kernel_max;
extern const char *kernel_mean;
extern const char *kernel_nll_loss;

class CUDAInitializer {
public:
  CUdevice cuDevice;
  CUcontext cuContext;
  
  CUDAInitializer() {
  // FIXME: doesn't work
    std::cout << "Initializing CUDA context\n";
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
  }
  
  ~CUDAInitializer() {
    std::cout << "Destroying CUDA context\n";
    CUDA_SAFE_CALL(cuCtxDestroy(cuContext));
  }
};  

#endif // TENSORLIBRARY_KERNELS_H
