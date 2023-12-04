#include "kernels.h"

const char *kernel_add = "                                              \n\
extern \"C\" __global__ void add(int max_size, float *x, float *y,      \n\
                                 float *out, int x_size, int y_size) {  \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < max_size; i += stride) {                      \n\
    out[i] = x[i % x_size] + y[i % y_size];                             \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_neg = "                                              \n\
extern \"C\" __global__ void neg(float *x, float *out, int x_size) {    \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    out[i] = -x[i];                                                     \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_mul = "                                              \n\
extern \"C\" __global__ void mul(int max_size, float *x, float *y,      \n\
                                 float *out, int x_size, int y_size) {  \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < max_size; i += stride) {                      \n\
    out[i] = x[i % x_size] * y[i % y_size];                             \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_div = "                                              \n\
extern \"C\" __global__ void div(int max_size, float *x, float *y,      \n\
                                 float *out, int x_size, int y_size) {  \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < max_size; i += stride) {                      \n\
    out[i] = x[i % x_size] / y[i % y_size];                             \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_log = "                                              \n\
extern \"C\" __global__ void log(float *x, float *out, int x_size) {    \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    out[i] = logf(x[i]);                                                \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_exp = "                                              \n\
extern \"C\" __global__ void exp(float *x, float *out, int x_size) {    \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    out[i] = expf(x[i]);                                                \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_sum = "                                              \n\
extern \"C\" __global__ void sum(float *x, float *out, int x_size) {    \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    atomicAdd(out, x[i]);                                               \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_max = "                                              \n\
extern \"C\" __global__ void max(float *x, float *out, int x_size) {    \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    atomicMax(out, x[i]);                                               \n\
  }                                                                     \n\
}                                                                       \n";


//  size_t n = x->shape.size();
//  size_t m = y->shape.size();
//  assert(n >= 1 && m >= 1);
//  assert(x->shape[n - 1] == y->shape[m - 2]);
//  size_t batch_size = 1;
//  for (size_t i = 0; i < n - 2; i++) {
//    assert(x->shape[i] == 1 || x->shape[i] == y->shape[i]);
//    batch_size *= x->shape[i];
//  }
//  size_t dim0 = x->shape[n - 2];
//  size_t dim1 = y->shape[m - 1];
//  size_t dim2 = y->shape[m - 2];
//      for (size_t i = 0; i < batch_size; i++) {
//        for (size_t j = 0; j < dim0; j++) {
//          for (size_t k = 0; k < dim1; k++) {
//            for (size_t l = 0; l < dim2; l++) {
//              result->data[i * dim0 * dim1 + j * dim1 + k] +=
//                  this->data[i * dim0 * dim2 + j * dim2 + l] *
//                  other->data[i * dim2 * dim1 + l * dim1 + k];
//            }
//          }
//        }
//      }

// batched matmul
const char *kernel_matmul = "                                                     \n\
extern \"C\" __global__ void matmul(int max_size, float *x, float *y,             \n\
                                    float *out, int x_size, int y_size, int dim0, \n\
                                    int dim1, int dim2) {                         \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  int stride = blockDim.x * gridDim.x;                                            \n\
  for (int i = index; i < max_size; i += stride) {                                \n\
    int batch = i / (dim0 * dim1 * dim2);                                         \n\
    int j = (i - batch * dim0 * dim1 * dim2) / (dim1 * dim2);                     \n\
    int k = (i - batch * dim0 * dim1 * dim2 - j * dim1 * dim2) / dim2;            \n\
    int l = i - batch * dim0 * dim1 * dim2 - j * dim1 * dim2 - k * dim2;          \n\
    out[i] = x[batch * dim0 * dim2 + j * dim2 + l] * y[batch * dim2 * dim1 + l * dim1 + k]; \n\
  }                                                                               \n\
}                                                                                 \n";

const char *kernel_relu = "                                             \n\
extern \"C\" __global__ void relu(float *x, float *out, int x_size) {   \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    out[i] = x[i] > 0 ? x[i] : 0;                                       \n\
  }                                                                     \n\
}                                                                       \n";

const char *kernel_nll_loss = "                                         \n\
extern \"C\" __global__ void nll_loss(float *x, float *y, float *out,   \n\
                                      int x_size, int y_size) {         \n\
  int index = blockIdx.x * blockDim.x + threadIdx.x;                    \n\
  int stride = blockDim.x * gridDim.x;                                  \n\
  for (int i = index; i < x_size; i += stride) {                        \n\
    out[i] = -x[i * y_size + static_cast<int>(y[i]+0.001)];             \n\
  }                                                                     \n\
}                                                                       \n";
