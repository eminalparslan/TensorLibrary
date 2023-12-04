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

extern "C" __global__ void _kernel_add_grad(int max_size, float *x_grad, float *y_grad,
                                 float *out_grad, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < max_size; i += stride) {
    x_grad[i % x_size] += out_grad[i];
    y_grad[i % y_size] += out_grad[i];
  }
}

void kernel_add_grad(int max_size, float *x_grad, float *y_grad, float *out_grad, int x_size, int y_size) {
  _kernel_add_grad<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x_grad, y_grad, out_grad, x_size, y_size);
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

extern "C" __global__ void _kernel_neg_grad(float *x_grad, float *out_grad, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] -= out_grad[i];
  }
}

void kernel_neg_grad(float *x_grad, float *out_grad, int x_size) {
  _kernel_neg_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x_size);
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

extern "C" __global__ void _kernel_mul_grad(int max_size, float *x_grad, float *y_grad,
                                 float *out_grad, float *x, float *y, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < max_size; i += stride) {
    x_grad[i % x_size] += out_grad[i] * y[i % y_size];
    y_grad[i % y_size] += out_grad[i] * x[i % x_size];
  }
}

void kernel_mul_grad(int max_size, float *x_grad, float *y_grad, float *out_grad, float *x, float *y, int x_size, int y_size) {
  _kernel_mul_grad<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x_grad, y_grad, out_grad, x, y, x_size, y_size);
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

extern "C" __global__ void _kernel_div_grad(int max_size, float *x_grad, float *y_grad,
                                 float *out_grad, float *x, float *y, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < max_size; i += stride) {
    x_grad[i % x_size] += out_grad[i] / y[i % y_size];
    float y2 = y[i % y_size] * y[i % y_size];
    y_grad[i % y_size] -= out_grad[i] * x[i % x_size] / y2;
  }
}

void kernel_div_grad(int max_size, float *x_grad, float *y_grad, float *out_grad, float *x, float *y, int x_size, int y_size) {
  _kernel_div_grad<<<NUM_BLOCKS, NUM_THREADS>>>(max_size, x_grad, y_grad, out_grad, x, y, x_size, y_size);
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

extern "C" __global__ void _kernel_log_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] += out_grad[i] / x[i];
  }
}

void kernel_log_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  _kernel_log_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x, x_size);
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

extern "C" __global__ void _kernel_exp_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] += out_grad[i] * x[i];
  }
}

void kernel_exp_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  _kernel_exp_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x, x_size);
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

extern "C" __global__ void _kernel_sum_grad(float *x_grad, float *out_grad, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] += out_grad[0];
  }
}

void kernel_sum_grad(float *x_grad, float *out_grad, int x_size) {
  _kernel_sum_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x_size);
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

extern "C" __global__ void _kernel_max_grad(float *x_grad, float *out_grad, float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] += out_grad[0] * (abs(x[i] - out[0]) < 0.001);
  }
}

void kernel_max_grad(float *x_grad, float *out_grad, float *x, float *out, int x_size) {
  _kernel_max_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x, out, x_size);
}

extern "C" __global__ void _kernel_matmul(float *x, float *y, float *out,
                                      int m, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < m * k; i += stride) {
    int row = i / k;
    int col = i % k;
    float sum = 0;
    for (int j = 0; j < n; j++) {
      sum += x[row * n + j] * y[j * k + col];
    }
    out[i] = sum;
  }
}

void kernel_matmul(float *x, float *y, float *out, int m, int n, int k) {
  _kernel_matmul<<<NUM_BLOCKS, NUM_THREADS>>>(x, y, out, m, n, k);
}

extern "C" __global__ void _kernel_matmul_grad(float *x_grad, float *y_grad, float *out_grad,
                                      float *x, float *y, int m, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < m * k; i += stride) {
    int row = i / k;
    int col = i % k;
    for (int j = 0; j < n; j++) {
      x_grad[row * n + j] += out_grad[i] * y[j * k + col];
      y_grad[j * k + col] += out_grad[i] * x[row * n + j];
    }
  }
}

void kernel_matmul_grad(float *x_grad, float *y_grad, float *out_grad, float *x, float *y, int m, int n, int k) {
  _kernel_matmul_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, y_grad, out_grad, x, y, m, n, k);
}

extern "C" __global__ void _kernel_relu(float *x, float *out, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < x_size; i += stride) {
    out[i] = x[i] * (x[i] > 0);
  }
}

void kernel_relu(float *x, float *out, int x_size) {
  _kernel_relu<<<NUM_BLOCKS, NUM_THREADS>>>(x, out, x_size);
}

extern "C" __global__ void _kernel_relu_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i] += out_grad[i] * (x[i] > 0);
  }
}

void kernel_relu_grad(float *x_grad, float *out_grad, float *x, int x_size) {
  _kernel_relu_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, out_grad, x, x_size);
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

extern "C" __global__ void _kernel_nll_loss_grad(float *x_grad, float *y_grad, float *out_grad,
                                      float *x, float *y, int x_size, int y_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x_grad[i * y_size + (int)(y[i]+0.001)] -= out_grad[i];
  }
}

void kernel_nll_loss_grad(float *x_grad, float *y_grad, float *out_grad, float *x, float *y, int x_size, int y_size) {
  _kernel_nll_loss_grad<<<NUM_BLOCKS, NUM_THREADS>>>(x_grad, y_grad, out_grad, x, y, x_size, y_size);
}

extern "C" __global__ void _kernel_SGD_step(float *x, float *x_grad, int x_size, float lr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < x_size; i += stride) {
    x[i] -= lr * x_grad[i];
  }
}

void kernel_SGD_step(float *x, float *x_grad, int x_size, float lr) {
  _kernel_SGD_step<<<NUM_BLOCKS, NUM_THREADS>>>(x, x_grad, x_size, lr);
}
