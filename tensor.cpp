//
// Created by Emin Arslan on 11/9/23.
//

#include "tensor.h"

std::shared_ptr<TensorImpl>
TensorImpl::add(const std::shared_ptr<TensorImpl> &other) {
  // TODO: check dimensions
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()});
  for (size_t i = 0; i < max_size; i++) {
    size_t this_index = i % this->size;
    size_t other_index = i % other->size;
    result->data[i] = this->data[this_index] + other->data[other_index];
  }
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
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = -this->data[i];
  }
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
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()});
  for (size_t i = 0; i < max_size; i++) {
    size_t this_index = i % this->size;
    size_t other_index = i % other->size;
    result->data[i] = this->data[this_index] * other->data[other_index];
  }
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
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = this->data[i] * other;
  }
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
  size_t max_size = std::max(this->size, other->size);
  auto result = std::make_shared<TensorImpl>(
      this->size == max_size ? this->shape : other->shape,
      std::initializer_list<TensorImpl *>{this, other.get()});
  for (size_t i = 0; i < max_size; i++) {
    size_t this_index = i % this->size;
    size_t other_index = i % other->size;
    result->data[i] = this->data[this_index] / other->data[other_index];
  }
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
  size_t n = this->shape.size();
  size_t m = other->shape.size();
  if (n == 1 && m == 1) {
    // vector-vector dot product
    assert(this->shape[0] == other->shape[0]);
    // calculate dot product
    float dot = 0;
    for (size_t i = 0; i < this->size; i++) {
      dot += this->data[i] * other->data[i];
    }
    auto result = std::make_shared<TensorImpl>(
        std::initializer_list<int>{1},
        std::initializer_list<TensorImpl *>{this, other.get()});
    result->data[0] = dot;
    result->grad_fn = [=]() {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result->grad[i] * other->data[i];
        other->grad[i] += result->grad[i] * this->data[i];
      }
    };
    return result;
  } else if (n == 2 && m == 2) {
    // matrix-matrix multiplication
    assert(this->shape[1] == other->shape[0]);
    int p = this->shape[0];
    int q = this->shape[1];
    int r = other->shape[1];
    auto result = std::shared_ptr<TensorImpl>(
        new TensorImpl({p, r}, {this, other.get()}));
    // TODO: optimize by blocking
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < r; j++) {
        float dot = 0;
        for (int k = 0; k < q; k++) {
          dot += this->data[i * q + k] * other->data[k * r + j];
        }
        result->data[i * r + j] = dot;
      }
    }
    result->grad_fn = [=]() {
      for (int i = 0; i < p; i++) {
        for (int j = 0; j < r; j++) {
          for (int k = 0; k < q; k++) {
            this->grad[i * q + k] +=
                result->grad[i * r + j] * other->data[k * r + j];
            other->grad[k * r + j] +=
                result->grad[i * r + j] * this->data[i * q + k];
          }
        }
      }
    };
    return result;
  } else if (n == 2 && m == 1) {
    // matrix-vector multiplication
    assert(this->shape[1] == other->shape[0]);
    int p = this->shape[0];
    int q = this->shape[1];
    auto result = std::make_shared<TensorImpl>(
        std::initializer_list<int>{p},
        std::initializer_list<TensorImpl *>{this, other.get()});

    for (int i = 0; i < p; i++) {
      float dot = 0;
      for (int k = 0; k < q; k++) {
        dot += this->data[i * q + k] * other->data[k];
      }
      result->data[i] = dot;
    }
    result->grad_fn = [=]() {
      for (int i = 0; i < p; i++) {
        for (int k = 0; k < q; k++) {
          this->grad[i * q + k] += result->grad[i] * other->data[k];
          other->grad[k] += result->grad[i] * this->data[i * q + k];
        }
      }
    };
    return result;
  } else if (n == 1 && m == 2) {
    // vector-matrix multiplication
    assert(this->shape[0] == other->shape[0]);
    int p = this->shape[0];
    int q = other->shape[1];
    auto result = std::make_shared<TensorImpl>(
        std::initializer_list<int>{q},
        std::initializer_list<TensorImpl *>{this, other.get()});
    for (int j = 0; j < q; j++) {
      float dot = 0;
      for (int k = 0; k < p; k++) {
        dot += this->data[k] * other->data[k * q + j];
      }
      result->data[j] = dot;
    }
    result->grad_fn = [=]() {
      for (int j = 0; j < q; j++) {
        for (int k = 0; k < p; k++) {
          this->grad[k] += result->grad[j] * other->data[k * q + j];
          other->grad[k * q + j] += result->grad[j] * this->data[k];
        }
      }
    };
    return result;
  } else if (n > 2) {
    // batched matrix multiplication

  } else if (m > 2) {
    // batched matrix multiplication

  } else {
    assert(false);
  }
  return nullptr;
}

// TODO: verify code
std::shared_ptr<TensorImpl> TensorImpl::transpose(size_t dim0, size_t dim1) {
  assert(dim0 < this->shape.size() && dim1 < this->shape.size());
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{this->shape[dim1], this->shape[dim0]},
      std::initializer_list<TensorImpl *>{this});
  for (int i = 0; i < this->shape[dim0]; i++) {
    for (int j = 0; j < this->shape[dim1]; j++) {
      result->data[j * this->shape[dim0] + i] =
          this->data[i * this->shape[dim1] + j];
    }
  }
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
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = this->data[i] > 0.0f ? this->data[i] : 0.0f;
  }
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] * (this->data[i] > 0.0f ? 1.0f : 0.0f);
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::sigmoid() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }
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
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = std::log(this->data[i]);
  }
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] / this->data[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::exp() {
  auto result = std::make_shared<TensorImpl>(
      this->shape, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[i] = std::exp(this->data[i]);
  }
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[i] * result->data[i];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::sum() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[0] += this->data[i];
  }
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[0];
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::max() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this});
  result->data[0] = this->data[0];
  for (size_t i = 1; i < this->size; i++) {
    result->data[0] = std::max(result->data[0], this->data[i]);
  }
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
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this});
  for (size_t i = 0; i < this->size; i++) {
    result->data[0] += this->data[i];
  }
  result->data[0] /= this->size;
  result->grad_fn = [=]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result->grad[0] / this->size;
    }
  };
  return result;
}

std::shared_ptr<TensorImpl> TensorImpl::argmax() {
  auto result = std::make_shared<TensorImpl>(
      std::initializer_list<int>{1}, std::initializer_list<TensorImpl *>{this});
  result->data[0] = 0;
  for (size_t i = 1; i < this->size; i++) {
    if (this->data[i] > this->data[(int)result->data[0]]) {
      result->data[0] = i;
    }
  }
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
  assert(this->shape.size() == 2 && other->shape.size() == 1);
  assert(this->shape[0] == other->shape[0]);
  int batch_size = this->shape[0];
  int num_classes = this->shape[1];
  auto result = std::make_shared<TensorImpl>(
      other->shape, std::initializer_list<TensorImpl *>{this, other.get()});
  for (int i = 0; i < batch_size; i++) {
    result->data[i] -=
        this->data[i * num_classes + (int)std::round(other->data[i])];
  }
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