#include "kernels.h"

extern "C" __global__ void _kernel_add(int max_size, float *x, float *y,
                                 float *out, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < max_size; i += stride) {
    out[i] = x[i % x_size] + y[i % y_size];
  }
}

void kernel_add(int max_size, float *x, float *y, float *out, int x_size, int y_size) {
  _kernel_add<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x, y, out, x_size, y_size);
}

extern "C" __global__ void _kernel_neg(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = -x[i];
  }
}

void kernel_neg(float *x, float *out, int x_size) {
  _kernel_neg<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

extern "C" __global__ void _kernel_mul(int max_size, float *x, float *y,
                                 float *out, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < max_size; i += stride) {
    out[i] = x[i % x_size] * y[i % y_size];
  }
}

void kernel_mul(int max_size, float *x, float *y, float *out, int x_size, int y_size) {
  _kernel_mul<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x, y, out, x_size, y_size);
}

extern "C" __global__ void _kernel_div(int max_size, float *x, float *y,
                                 float *out, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < max_size; i += stride) {
    out[i] = x[i % x_size] / y[i % y_size];
  }
}



void kernel_div(int max_size, float *x, float *y, float *out, int x_size, int y_size) {
  _kernel_div<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x, y, out, x_size, y_size);
}

extern "C" __global__ void _kernel_log(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = logf(x[i]);
  }
}

void kernel_log(float *x, float *out, int x_size) {
  _kernel_log<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

extern "C" __global__ void _kernel_exp(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = expf(x[i]);
  }
}

void kernel_exp(float *x, float *out, int x_size) {
  _kernel_exp<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

extern "C" __global__ void _kernel_sum(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    atomicAdd(out, x[i]);
  }
}

void kernel_sum(float *x, float *out, int x_size) {
  _kernel_sum<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

// https://stackoverflow.com/a/17401122/8786742
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

extern "C" __global__ void _kernel_max(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // set out to -inf
  if (index == 0) {
    *out = -INFINITY;
  }

  __syncthreads();

  for (int i = index; i < x_size; i += stride) {
    atomicMax(out, x[i]);
  }
}

void kernel_max(float *x, float *out, int x_size) {
  _kernel_max<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

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
extern "C" __global__ void _kernel_matmul(int max_size, float *x, float *y,
                                    float *out, int x_size, int y_size, int dim0,
                                    int dim1, int dim2) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < max_size; i += stride) {
    int batch = i / (dim0 * dim1 * dim2);
    int j = (i - batch * dim0 * dim1 * dim2) / (dim1 * dim2);
    int k = (i - batch * dim0 * dim1 * dim2 - j * dim1 * dim2) / dim2;
    int l = i - batch * dim0 * dim1 * dim2 - j * dim1 * dim2 - k * dim2;
    out[i] = x[batch * dim0 * dim2 + j * dim2 + l] * y[batch * dim2 * dim1 + l * dim1 + k];
  }
}

void kernel_matmul(int max_size, float *x, float *y, float *out, int x_size, int y_size, int dim0,
                   int dim1, int dim2) {
  _kernel_matmul<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x, y, out, x_size, y_size, dim0, dim1, dim2);
}

extern "C" __global__ void _kernel_relu(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = x[i] > 0 ? x[i] : 0;
  }
}

void kernel_relu(float *x, float *out, int x_size) {
  _kernel_relu<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

extern "C" __global__ void _kernel_nll_loss(float *x, float *y, float *out,
                                      int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = -x[i * y_size + (int)(y[i]+0.001)];
  }
}

void kernel_nll_loss(float *x, float *y, float *out, int x_size, int y_size) {
  _kernel_nll_loss<<<NUM_BLOCKS, NUM_THREADS>>>(x, y, out, x_size, y_size);
}
