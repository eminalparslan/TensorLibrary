//
// Created by Emin Arslan on 11/9/23.
//

#include "tensor.h"

void launch_kernel(const char *kernel, const char *name, void *args[]) {
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel, name, 0, nullptr, nullptr));
  const char *opts[] = {"--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  std::cout << "Kernel compilation log:" << std::endl;
  if (logSize > 1) {
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    std::cout << log << std::endl;
    delete[] log;
  }

  if (compileResult != NVRTC_SUCCESS) {
    throw std::runtime_error("Kernel compilation failed");
  }
  
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  CUmodule module;
  CUfunction kernel_func;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_func, module, name));
  
  CUDA_SAFE_CALL(cuLaunchKernel(kernel_func, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1, 0, nullptr, args, 0));
  // TODO: synchronize lazily
  CUDA_SAFE_CALL(cuCtxSynchronize());
  CUDA_SAFE_CALL(cuModuleUnload(module));
}

std::shared_ptr<TensorImpl>
TensorImpl::add(const std::shared_ptr<TensorImpl> &other) {
  // TODO: check dimensions
  assert(this->backend == other->backend);
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  // CUDA variant: allocate beforehand, launch kernel in calc_fn
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        result->data[i] = this->data[this_index] + other->data[other_index];
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {(void*)&max_size, &this->cuda_data, &other->cuda_data, &result->cuda_data, &this->size, &other->size};
      launch_kernel(kernel_add, "add", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < max_size; i++) {
      size_t this_index = i % this->size;
      size_t other_index = i % other->size;
      this->grad[this_index] += result->grad[i];
      other->grad[other_index] += result->grad[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::neg() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        result->data[i] = -this->data[i];
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_neg, "neg", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] -= result->grad[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::sub(const std::shared_ptr<TensorImpl> &other) {
  return this->add(other->neg());
}

std::shared_ptr<TensorImpl>
TensorImpl::mul(const std::shared_ptr<TensorImpl> &other) {
  // TODO: check dimensions
  assert(this->backend == other->backend);
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        result->data[i] = this->data[this_index] * other->data[other_index];
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {(void*)&max_size, &this->cuda_data, &other->cuda_data, &result->cuda_data, &this->size, &other->size};
      launch_kernel(kernel_mul, "mul", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < max_size; i++) {
      size_t this_index = i % this->size;
      size_t other_index = i % other->size;
      this->grad[this_index] += result->grad[i] * other->data[other_index];
      other->grad[other_index] += result->grad[i] * this->data[this_index];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::mul(float other) {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    assert(result->backend == Backend::CPU);
    for (size_t i = 0; i < this->size; i++) {
      result->data[i] = this->data[i] * other;
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] * other;
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::div(const std::shared_ptr<TensorImpl> &other) {
  // TODO: check dimensions
  assert(this->backend == other->backend);
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  result->calc_fn = [=]() {
    if (backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        result->data[i] = this->data[this_index] / other->data[other_index];
      }
    } else if (backend == Backend::CUDA) {
      void *args[] = {(void*)&max_size, &this->cuda_data, &other->cuda_data, &result->cuda_data, &this->size, &other->size};
      launch_kernel(kernel_div, "div", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < max_size; i++) {
      size_t this_index = i % this->size;
      size_t other_index = i % other->size;
      this->grad[this_index] += result->grad[i] / other->data[other_index];
      other->grad[other_index] -=
          result->grad[i] * this->data[this_index] /
          (other->data[other_index] * other->data[other_index]);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::matmul(const std::shared_ptr<TensorImpl> &other) {
  // TODO: test this function
  // TODO: understand order of dimensions in shape vs PyTorch
  assert(this->backend == other->backend);
  size_t n = this->shape.size();
  size_t m = other->shape.size();
  assert(n >= 1 && m >= 1);
  assert(this->shape[n - 1] == other->shape[m - 2]);
  size_t batch_size = 1;
  for (size_t i = 0; i < n - 2; i++) {
    assert(this->shape[i] == 1 || this->shape[i] == other->shape[i]);
    batch_size *= this->shape[i];
  }
  size_t dim0 = this->shape[n - 2];
  size_t dim1 = other->shape[m - 1];
  size_t dim2 = other->shape[m - 2];
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{(int)batch_size, (int)dim0, (int)dim1},
      std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < dim0; j++) {
          for (size_t k = 0; k < dim1; k++) {
            for (size_t l = 0; l < dim2; l++) {
              result->data[i * dim0 * dim1 + j * dim1 + k] +=
                  this->data[i * dim0 * dim2 + j * dim2 + l] *
                  other->data[i * dim2 * dim1 + l * dim1 + k];
            }
          }
        }
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {(void*)&batch_size, (void*)&dim0, (void*)&dim1, (void*)&dim2, &this->cuda_data, &other->cuda_data, &result->cuda_data};
      launch_kernel(kernel_matmul, "matmul", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < dim0; j++) {
        for (size_t k = 0; k < dim1; k++) {
          for (size_t l = 0; l < dim2; l++) {
            this->grad[i * dim0 * dim2 + j * dim2 + l] +=
                result->grad[i * dim0 * dim1 + j * dim1 + k] *
                other->data[i * dim2 * dim1 + l * dim1 + k];
            other->grad[i * dim2 * dim1 + l * dim1 + k] +=
                result->grad[i * dim0 * dim1 + j * dim1 + k] *
                this->data[i * dim0 * dim2 + j * dim2 + l];
          }
        }
      }
    }
  };
  return result;
}

// TODO: verify code
std::shared_ptr<TensorImpl> TensorImpl::transpose(size_t dim0, size_t dim1) {
  assert(dim0 < this->shape.size() && dim1 < this->shape.size());
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{this->shape[dim1], this->shape[dim0]},
      std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    assert(result->backend == Backend::CPU);
    for (int i = 0; i < this->shape[dim0]; i++) {
      for (int j = 0; j < this->shape[dim1]; j++) {
        result->data[j * this->shape[dim0] + i] =
            this->data[i * this->shape[dim1] + j];
      }
    }
  };
  result->grad_fn = [=]() {
    for (int i = 0; i < this->shape[dim0]; i++) {
      for (int j = 0; j < this->shape[dim1]; j++) {
        this->grad[i * this->shape[dim1] + j] +=
            result->grad[j * this->shape[dim0] + i];
      }
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::relu() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        result->data[i] = this->data[i] > 0.0f ? this->data[i] : 0.0f;
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_relu, "relu", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] * (this->data[i] > 0.0f ? 1.0f : 0.0f);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::sigmoid() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    assert(result->backend == Backend::CPU);
    for (size_t i = 0; i < this->size; i++) {
      result->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      float s = 1.0f / (1.0f + std::exp(-this->data[i]));
      this->grad[i] += result->grad[i] * s * (1.0f - s);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::log() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        result->data[i] = std::log(this->data[i]);
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_log, "log", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] / this->data[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::exp() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        result->data[i] = std::exp(this->data[i]);
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_exp, "exp", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] * result->data[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::sum() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        result->data[0] += this->data[i];
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_sum, "sum", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[0];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::max() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      result->data[0] = this->data[0];
      for (size_t i = 1; i < this->size; i++) {
        result->data[0] = std::max(result->data[0], this->data[i]);
      }
    } else if (result->backend == Backend::CUDA) {
      void *args[] = {&this->cuda_data, &result->cuda_data, (void*)&this->size};
      launch_kernel(kernel_max, "max", args);
    }
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      if (this->data[i] == result->data[0]) {
        this->grad[i] += result->grad[0];
      }
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::mean() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    assert(result->backend == Backend::CPU);
    for (size_t i = 0; i < this->size; i++) {
      result->data[0] += this->data[i];
    }
    result->data[0] /= this->size;
  };
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[0] / this->size;
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::argmax() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this}, this->backend);
  result->calc_fn = [=]() {
    assert(result->backend == Backend::CPU);
    result->data[0] = 0;
    for (size_t i = 1; i < this->size; i++) {
      if (this->data[i] > this->data[(int)result->data[0]]) {
        result->data[0] = i;
      }
    }
  };
  result->grad_fn = [=]() {
    this->grad[(int)result->data[0]] += result->grad[0];
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::softmax() {
  auto shifted = this->sub(this->max());
  auto exp = shifted->exp();
  return exp->div(exp->sum());
}

std::shared_ptr<TensorImpl> TensorImpl::log_softmax() {
  // TODO: optimized implementation w/o softmax
  return this->softmax()->log();
}

std::shared_ptr<TensorImpl>
TensorImpl::nll_loss(const std::shared_ptr<TensorImpl> &other) {
  assert(this->backend == other->backend);
  assert(this->shape.size() == 2 && other->shape.size() == 1);
  assert(this->shape[0] == other->shape[0]);
  int batch_size = this->shape[0];
  int num_classes = this->shape[1];
  auto result = std::make_shared<TensorImpl>(
      other->shape, std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  result->calc_fn = [=]() {
    for (int i = 0; i < batch_size; i++) {
      result->data[i] = -this->data[i * num_classes + (int)std::round(other->data[i])];
    }
  };
  result->grad_fn = [=]() {
    for (int i = 0; i < batch_size; i++) {
      this->grad[i * num_classes + (int)std::round(other->data[i])] -=
          result->grad[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::cross_entropy_loss(const std::shared_ptr<TensorImpl> &other) {
  return this->log_softmax()->nll_loss(other);
}