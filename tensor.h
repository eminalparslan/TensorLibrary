//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_TENSOR_H
#define TENSORLIBRARY_TENSOR_H

#include <functional>
#include <iostream>
#include <vector>

class Tensor {
public:
  int *data{nullptr};
  int *grad{nullptr};
  std::vector<int> shape;
  size_t size;

  // FIXME: might not need initializer_list
  Tensor(std::initializer_list<int> list, std::vector<int> shape)
      : shape(shape) {
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new int[this->size];
    this->grad = new int[this->size];
    int i = 0;
    for (auto &item : list) {
      this->data[i] = item;
      this->grad[i] = 0;
      i++;
    }
  }

  Tensor(std::initializer_list<int> list, std::initializer_list<int> shape)
      : Tensor(list, std::vector(shape)) {}

  Tensor(std::initializer_list<int> list)
      : Tensor(list, {static_cast<int>(list.size())}) {}

  ~Tensor() {
    delete[] data;
    delete[] grad;
  }

  // TODO: consider carrying gradients through copy constructor

  void backward() {
    // TODO: use topological sort instead to deal with DAGs (right now it only
    // works for trees)
    if (grad_fn != nullptr) {
      grad_fn();
      if (lhs != nullptr) {
        lhs->backward();
      }
      if (rhs != nullptr) {
        rhs->backward();
      }
    }
  }

  Tensor operator+(Tensor &other);
  Tensor operator*(Tensor &other);
  Tensor operator-();
  Tensor operator-(Tensor &other);
  Tensor operator/(Tensor &other);

  Tensor matmul(Tensor &other);
  Tensor transpose();

  Tensor relu();

private:
  Tensor *lhs{nullptr};
  Tensor *rhs{nullptr};
  std::function<void()> grad_fn{nullptr};

  Tensor(std::vector<int> shape, Tensor *lhs, Tensor *rhs) : shape(shape) {
    this->lhs = lhs;
    this->rhs = rhs;
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new int[this->size];
    this->grad = new int[this->size];
    for (size_t i = 0; i < this->size; i++) {
      this->data[i] = 0;
      this->grad[i] = 0;
    }
  }

  Tensor(std::vector<int> shape, Tensor *lhs) : shape(shape) {
    Tensor(shape, lhs, nullptr);
  }
};

#endif // TENSORLIBRARY_TENSOR_H
