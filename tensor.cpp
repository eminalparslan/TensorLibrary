//
// Created by Emin Arslan on 11/9/23.
//

#include "tensor.h"

std::shared_ptr<TensorImpl>
TensorImpl::add(const std::shared_ptr<TensorImpl> &other) {
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
        result->data[i] = this->data[this_index] + other->data[other_index];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_add(max_size, this->cuda_data, other->cuda_data, result->cuda_data, this->size, other->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        this->grad[this_index] += result->grad[i];
        other->grad[other_index] += result->grad[i];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_add_grad(max_size, this->cuda_grad, other->cuda_grad, result->cuda_grad, this->size, other->size);
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
      kernel_neg(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] -= result->grad[i];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_neg_grad(this->cuda_grad, result->cuda_grad, this->size);
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
      kernel_mul(max_size, this->cuda_data, other->cuda_data, result->cuda_data, this->size, other->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        this->grad[this_index] += result->grad[i] * other->data[other_index];
        other->grad[other_index] += result->grad[i] * this->data[this_index];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_mul_grad(max_size, this->cuda_grad, other->cuda_grad, result->cuda_grad, this->cuda_data, other->cuda_data, this->size, other->size);
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
    assert(result->backend == Backend::CPU);
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
      kernel_div(max_size, this->cuda_data, other->cuda_data, result->cuda_data, this->size, other->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < max_size; i++) {
        size_t this_index = i % this->size;
        size_t other_index = i % other->size;
        this->grad[this_index] += result->grad[i] / other->data[other_index];
        other->grad[other_index] -=
            result->grad[i] * this->data[this_index] /
            (other->data[other_index] * other->data[other_index]);
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_div_grad(max_size, this->cuda_grad, other->cuda_grad, result->cuda_grad, this->cuda_data, other->cuda_data, this->size, other->size);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::matmul(const std::shared_ptr<TensorImpl> &other) {
  // matrix-matrix, matrix-vector, vector-matrix, vector-vector
  assert(this->backend == other->backend);
  assert(this->shape.size() == 2 && other->shape.size() == 2);
  assert(this->shape[1] == other->shape[0]);
  int m = this->shape[0];
  int n = this->shape[1];
  int k = other->shape[1];
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{m, k},
      std::initializer_list<TensorImpl *>{this, other.get()}, this->backend);
  result->calc_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
          for (int l = 0; l < n; l++) {
            result->data[i * k + j] +=
                this->data[i * n + l] * other->data[l * k + j];
          }
        }
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_matmul(this->cuda_data, other->cuda_data, result->cuda_data, m, n, k);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
          for (int l = 0; l < n; l++) {
            this->grad[i * n + l] +=
                result->grad[i * k + j] * other->data[l * k + j];
            other->grad[l * k + j] +=
                result->grad[i * k + j] * this->data[i * n + l];
          }
        }
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_matmul_grad(this->cuda_grad, other->cuda_grad, result->cuda_grad, this->cuda_data, other->cuda_data, m, n, k);
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
    assert(result->backend == Backend::CPU);
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
      kernel_relu(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result->grad[i] * (this->data[i] > 0.0f ? 1.0f : 0.0f);
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_relu_grad(this->cuda_grad, result->cuda_grad, this->cuda_data, this->size);
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
    assert(result->backend == Backend::CPU);
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
      kernel_log(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result->grad[i] / this->data[i];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_log_grad(this->cuda_grad, result->cuda_grad, this->cuda_data, this->size);
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
      kernel_exp(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result->grad[i] * result->data[i];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_exp_grad(this->cuda_grad, result->cuda_grad, result->cuda_data, this->size);
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
      kernel_sum(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result->grad[0];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_sum_grad(this->cuda_grad, result->cuda_grad, this->size);
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
      kernel_max(this->cuda_data, result->cuda_data, this->size);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (size_t i = 0; i < this->size; i++) {
        if (this->data[i] == result->data[0]) {
          this->grad[i] += result->grad[0];
        }
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_max_grad(this->cuda_grad, result->cuda_grad, this->cuda_data, result->cuda_data, this->size);
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
    assert(result->backend == Backend::CPU);
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
    assert(result->backend == Backend::CPU);
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
    if (result->backend == Backend::CPU) {
      for (int i = 0; i < batch_size; i++) {
        result->data[i] = -this->data[i * num_classes + (int)std::round(other->data[i])];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_nll_loss(this->cuda_data, other->cuda_data, result->cuda_data, batch_size, num_classes);
    }
  };
  result->grad_fn = [=]() {
    if (result->backend == Backend::CPU) {
      for (int i = 0; i < batch_size; i++) {
        this->grad[i * num_classes + (int)std::round(other->data[i])] -=
            result->grad[i];
      }
    } else if (result->backend == Backend::CUDA) {
      kernel_nll_loss_grad(this->cuda_grad, other->cuda_grad, result->cuda_grad, this->cuda_data, other->cuda_data, batch_size, num_classes);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl>
TensorImpl::cross_entropy_loss(const std::shared_ptr<TensorImpl> &other) {
  return this->log_softmax()->nll_loss(other);
}