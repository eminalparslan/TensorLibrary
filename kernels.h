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

void kernel_add(int max_size, float *x, float *y, float *out, int x_size, int y_size);
void kernel_neg(float *x, float *out, int x_size);
void kernel_mul(int max_size, float *x, float *y, float *out, int x_size, int y_size);
void kernel_div(int max_size, float *x, float *y, float *out, int x_size, int y_size);
void kernel_log(float *x, float *out, int x_size);
void kernel_exp(float *x, float *out, int x_size);
void kernel_sum(float *x, float *out, int x_size);
void kernel_max(float *x, float *out, int x_size);
void kernel_matmul(int max_size, float *x, float *y, float *out, int x_size, int y_size, int dim0,
                   int dim1, int dim2);
void kernel_relu(float *x, float *out, int x_size);
void kernel_nll_loss(float *x, float *y, float *out, int x_size, int y_size);

#endif // TENSORLIBRARY_KERNELS_H
