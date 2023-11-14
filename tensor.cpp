//
// Created by Emin Arslan on 11/9/23.
//

#include "tensor.h"

Tensor Tensor::operator+(Tensor &other) {
  assert(this->shape == other.shape);
  Tensor result(this->shape, this, &other);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = this->data[i] + other.data[i];
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result.grad[i];
      other.grad[i] += result.grad[i];
    }
  };
  return result;
}

Tensor Tensor::operator-() {
  Tensor result(this->shape, this);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = -this->data[i];
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] -= result.grad[i];
    }
  };
  return result;
}

Tensor Tensor::operator-(Tensor &other) {
  assert(this->shape == other.shape);
  Tensor result(this->shape, this, &other);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = this->data[i] - other.data[i];
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result.grad[i];
      other.grad[i] -= result.grad[i];
    }
  };
  return result;
}

Tensor Tensor::operator*(Tensor &other) {
  assert(this->shape == other.shape);
  Tensor result(this->shape, this, &other);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = this->data[i] * other.data[i];
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result.grad[i] * other.data[i];
      other.grad[i] += result.grad[i] * this->data[i];
    }
  };
  return result;
}

Tensor Tensor::operator/(Tensor &other) {
  assert(this->shape == other.shape);
  Tensor result(this->shape, this, &other);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = this->data[i] / other.data[i];
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result.grad[i] / other.data[i];
      other.grad[i] -=
          result.grad[i] * this->data[i] / (other.data[i] * other.data[i]);
    }
  };
  return result;
}

Tensor Tensor::matmul(Tensor &other) {
  size_t n = this->shape.size();
  size_t m = other.shape.size();
  if (n == 1 && m == 1) {
    // vector-vector dot product
    assert(this->shape[0] == other.shape[0]);
    // calculate dot product
    int dot = 0;
    for (size_t i = 0; i < this->size; i++) {
      dot += this->data[i] * other.data[i];
    }
    Tensor result({1}, this, &other);
    result.data[0] = dot;
    result.grad_fn = [&]() {
      for (size_t i = 0; i < this->size; i++) {
        this->grad[i] += result.grad[i] * other.data[i];
        other.grad[i] += result.grad[i] * this->data[i];
      }
    };
    return result;
  } else if (n == 2 && m == 2) {
    // matrix-matrix multiplication
    assert(this->shape[1] == other.shape[0]);
    int p = this->shape[0];
    int q = this->shape[1];
    int r = other.shape[1];
    Tensor result({p, r}, this, &other);
    // TODO: optimize by blocking
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < r; j++) {
        int dot = 0;
        for (int k = 0; k < q; k++) {
          dot += this->data[i * q + k] * other.data[k * r + j];
        }
        result.data[i * r + j] = dot;
      }
    }
    result.grad_fn = [p, r, q, this, &other, &result]() {
      for (int i = 0; i < p; i++) {
        for (int j = 0; j < r; j++) {
          for (int k = 0; k < q; k++) {
            this->grad[i * q + k] +=
                result.grad[i * r + j] * other.data[k * r + j];
            other.grad[k * r + j] +=
                result.grad[i * r + j] * this->data[i * q + k];
          }
        }
      }
    };
    return result;
  } else if (n == 2 && m == 1) {
    // matrix-vector multiplication
    assert(this->shape[1] == other.shape[0]);
    int p = this->shape[0];
    int q = this->shape[1];
    Tensor result({p}, this, &other);

    for (int i = 0; i < p; i++) {
      int dot = 0;
      for (int k = 0; k < q; k++) {
        dot += this->data[i * q + k] * other.data[k];
      }
      result.data[i] = dot;
    }
    result.grad_fn = [p, q, this, &other, &result]() {
      for (int i = 0; i < p; i++) {
        for (int k = 0; k < q; k++) {
          this->grad[i * q + k] += result.grad[i] * other.data[k];
          other.grad[k] += result.grad[i] * this->data[i * q + k];
        }
      }
    };
    return result;
  } else {
    assert(false);
  }
}

Tensor Tensor::relu() {
  Tensor result(this->shape, this);
  for (size_t i = 0; i < this->size; i++) {
    result.data[i] = this->data[i] > 0 ? this->data[i] : 0;
  }
  result.grad_fn = [&]() {
    for (size_t i = 0; i < this->size; i++) {
      this->grad[i] += result.grad[i] * (this->data[i] > 0 ? 1 : 0);
    }
  };
  return result;
}

