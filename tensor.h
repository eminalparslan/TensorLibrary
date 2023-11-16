//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_TENSOR_H
#define TENSORLIBRARY_TENSOR_H

#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

class TensorImpl {
private:
  std::unordered_set<TensorImpl *> children;
  std::function<void()> grad_fn{nullptr};

public:
  int *data{nullptr};
  int *grad{nullptr};
  std::vector<int> shape;
  size_t size;

  TensorImpl(int num, std::vector<int> shape) : shape(shape) {
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new int[this->size];
    this->grad = new int[this->size];
    for (size_t i = 0; i < this->size; i++) {
      this->data[i] = num;
      this->grad[i] = 0;
    }
  }

  TensorImpl(std::vector<int> list, std::vector<int> shape) : shape(shape) {
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

  TensorImpl(std::vector<int> list)
      : TensorImpl(list, {static_cast<int>(list.size())}) {}

  TensorImpl(std::vector<int> shape,
             std::initializer_list<TensorImpl *> children)
      : children(children), shape(shape) {
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

  ~TensorImpl() {
    delete[] data;
    delete[] grad;
  }

  // TODO: consider carrying gradients through copy constructor

  void backward() {
    for (size_t i = 0; i < size; i++) {
      grad[i] = 1;
    }

    std::vector<TensorImpl *> toposort;
    std::unordered_set<TensorImpl *> visited;

    std::function<void(TensorImpl *)> dfs = [&](TensorImpl *node) {
      if (visited.find(node) != visited.end() || node->grad_fn == nullptr) {
        return;
      }
      visited.insert(node);
      for (auto &child : node->children) {
        dfs(child);
      }
      toposort.push_back(node);
    };

    dfs(this);

    for (auto it = toposort.rbegin(); it != toposort.rend(); ++it) {
      (*it)->grad_fn();
    }
  }

  std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> mul(std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> neg();
  std::shared_ptr<TensorImpl> sub(std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> div(std::shared_ptr<TensorImpl> &other);

  std::shared_ptr<TensorImpl> matmul(std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> transpose();

  std::shared_ptr<TensorImpl> relu();
};

class Tensor {
private:

  Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}

public:
  std::shared_ptr<TensorImpl> impl;
  // TODO: check if copy constructor is called
  Tensor(int num, std::vector<int> shape) : impl(new TensorImpl(num, shape)) {}
  Tensor(std::vector<int> list, std::vector<int> shape)
      : impl(new TensorImpl(list, shape)) {}
  Tensor(std::vector<int> list) : impl(new TensorImpl(list)) {}

  size_t size() { return impl->size; }

  void backward() { impl->backward(); }

  void print() {
    std::cout << "[";
    for (size_t i = 0; i < impl->size; i++) {
      std::cout << impl->data[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  void printGrad() {
    std::cout << "[";
    for (size_t i = 0; i < impl->size; i++) {
      std::cout << impl->grad[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  Tensor operator+(Tensor &other) { return Tensor(impl->add(other.impl)); }
  Tensor operator*(Tensor &other) { return Tensor(impl->mul(other.impl)); }
  Tensor operator-() { return Tensor(impl->neg()); }
  Tensor operator-(Tensor &other) { return Tensor(impl->sub(other.impl)); }
  Tensor operator/(Tensor &other) { return Tensor(impl->div(other.impl)); }
  Tensor matmul(Tensor &other) { return Tensor(impl->matmul(other.impl)); }
  Tensor transpose() { return Tensor(impl->transpose()); }
  Tensor relu() { return Tensor(impl->relu()); }
};

#endif // TENSORLIBRARY_TENSOR_H
