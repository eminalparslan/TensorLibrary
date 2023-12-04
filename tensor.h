//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_TENSOR_H
#define TENSORLIBRARY_TENSOR_H

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

namespace nn {
class SGD;
}

// TODO: template for other types
// TODO: requires_grad
// TODO: cpu, cuda, etc. flags
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
private:
  std::unordered_set<TensorImpl *> children;
  // TODO: Test theory of keeping out of scope Tensors (use in intermediate
  // calculations) alive with lambda by copying Tensors with lambda and
  // checking reference counts.
  std::function<void()> calc_fn{nullptr};
  std::function<void()> grad_fn{nullptr};

public:
  float *data{nullptr};
  float *grad{nullptr};
  std::vector<int> shape;
  size_t size;
  bool realized{false};

  TensorImpl(float num, std::vector<int> shape) : shape(shape) {
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new float[this->size];
    this->grad = new float[this->size];
    for (size_t i = 0; i < this->size; i++) {
      this->data[i] = num;
      this->grad[i] = 0.0f;
    }
  }

  TensorImpl(std::vector<float> list, std::vector<int> shape) : shape(shape) {
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new float[this->size];
    this->grad = new float[this->size];
    int i = 0;
    for (auto &item : list) {
      this->data[i] = item;
      this->grad[i] = 0.0f;
      i++;
    }
  }

  TensorImpl(std::vector<float> list)
      : TensorImpl(list, {static_cast<int>(list.size())}) {}

  TensorImpl(std::vector<int> shape,
             std::initializer_list<TensorImpl *> children)
      : children(children), shape(shape) {
    this->size = 1;
    for (auto &item : shape) {
      this->size *= item;
    }
    this->data = new float[this->size];
    this->grad = new float[this->size];
    for (size_t i = 0; i < this->size; i++) {
      this->data[i] = 0.0f;
      this->grad[i] = 0.0f;
    }
  }

  ~TensorImpl() {
    delete[] data;
    delete[] grad;
  }

  // TODO: consider carrying gradients through copy constructor

  void topo_sort(std::unordered_set<TensorImpl *> &visited,
                 std::vector<TensorImpl *> &toposort) {
    if (visited.find(this) != visited.end()) {
      return;
    }
    visited.insert(this);
    for (auto &child : children) {
      child->topo_sort(visited, toposort);
    }
    toposort.push_back(this);
  }

  void backward() {
    for (size_t i = 0; i < size; i++) {
      grad[i] = 1.0f;
    }

    std::vector<TensorImpl *> toposort;
    std::unordered_set<TensorImpl *> visited;
    topo_sort(visited, toposort);

    for (auto it = toposort.rbegin(); it != toposort.rend(); ++it) {
      if ((*it)->grad_fn != nullptr) (*it)->grad_fn();
    }
  }

  void forward() {
    std::vector<TensorImpl *> toposort;
    std::unordered_set<TensorImpl *> visited;

    topo_sort(visited, toposort);

    for (auto &item : toposort) {
      if (item->calc_fn != nullptr) item->calc_fn();
    }
  }

  void zero_grad() {
    for (size_t i = 0; i < size; i++) {
      grad[i] = 0.0f;
    }
  }

  std::shared_ptr<TensorImpl> add(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> mul(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> mul(float other);
  std::shared_ptr<TensorImpl> neg();
  std::shared_ptr<TensorImpl> sub(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> div(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> matmul(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> transpose(size_t dim0, size_t dim1);
  std::shared_ptr<TensorImpl> relu();
  std::shared_ptr<TensorImpl> sigmoid();
  std::shared_ptr<TensorImpl> log();
  std::shared_ptr<TensorImpl> exp();
  std::shared_ptr<TensorImpl> sum();
  std::shared_ptr<TensorImpl> max();
  std::shared_ptr<TensorImpl> mean();
  std::shared_ptr<TensorImpl> argmax();
  std::shared_ptr<TensorImpl> softmax();
  std::shared_ptr<TensorImpl> log_softmax();
  std::shared_ptr<TensorImpl> nll_loss(const std::shared_ptr<TensorImpl> &other);
  std::shared_ptr<TensorImpl> cross_entropy_loss(const std::shared_ptr<TensorImpl> &other);

  friend nn::SGD;
};

class Tensor {
private:
  std::shared_ptr<TensorImpl> impl;
  // FIXME: is this needed?
  Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}

public:
  // TODO: check if copy constructor is called
  Tensor(float num, std::vector<int> shape) : impl(new TensorImpl(num, shape)) {
    impl->realized = true;
  }
  Tensor(std::vector<float> list, std::vector<int> shape)
      : impl(new TensorImpl(list, shape)) {
    impl->realized = true;
  }
  Tensor(std::vector<float> list) : impl(new TensorImpl(list)) {
    impl->realized = true;
  }

  // TODO: copy and assignment

  size_t size() { return impl->size; }
  std::vector<int> shape() { return impl->shape; }

  void backward() { impl->backward(); }
  Tensor &forward() {
    impl->forward();
    return *this;
  }
  Tensor &operator()() {
    impl->forward();
    return *this;
  }
  void zero_grad() { impl->zero_grad(); }

  /*Tensor grad() {*/
  /*  Tensor result = Tensor(0.0f, impl->shape);*/
  /*  for (size_t i = 0; i < impl->size; i++) {*/
  /*    result.impl->data[i] = impl->grad[i];*/
  /*  }*/
  /*  return result;*/
  /*}*/

  float item() {
    if (impl->size != 1) {
      throw std::runtime_error("item() only supported for size 1 Tensors");
    }
    return impl->data[0];
  }

  void print() {
    std::cout << "[";
    for (size_t i = 0; i < impl->size; i++) {
      std::cout << impl->data[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  void print_grad() {
    std::cout << "[";
    for (size_t i = 0; i < impl->size; i++) {
      std::cout << impl->grad[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  Tensor operator+(const Tensor &other) { return Tensor(impl->add(other.impl)); }
  Tensor operator*(const Tensor &other) { return Tensor(impl->mul(other.impl)); }
  Tensor operator*(float other) { return Tensor(impl->mul(other)); }
  Tensor operator-() { return Tensor(impl->neg()); }
  Tensor operator-(const Tensor &other) { return Tensor(impl->sub(other.impl)); }
  Tensor operator/(const Tensor &other) { return Tensor(impl->div(other.impl)); }
  Tensor matmul(const Tensor &other) { return Tensor(impl->matmul(other.impl)); }
  Tensor transpose(size_t dim0, size_t dim1) {
    return Tensor(impl->transpose(dim0, dim1));
  }
  Tensor relu() { return Tensor(impl->relu()); }
  Tensor sigmoid() { return Tensor(impl->sigmoid()); }
  Tensor log() { return Tensor(impl->log()); }
  Tensor exp() { return Tensor(impl->exp()); }
  Tensor sum() { return Tensor(impl->sum()); }
  Tensor max() { return Tensor(impl->max()); }
  Tensor mean() { return Tensor(impl->mean()); }
  Tensor argmax() { return Tensor(impl->argmax()); }
  Tensor softmax() { return Tensor(impl->softmax()); }
  Tensor log_softmax() { return Tensor(impl->log_softmax()); }
  Tensor nll_loss(const Tensor &other) { return Tensor(impl->nll_loss(other.impl)); }
  Tensor cross_entropy_loss(const Tensor &other) {
    return Tensor(impl->cross_entropy_loss(other.impl));
  }

  // FIXME: these are not in-place, and grads are not propagated?
  Tensor &operator+=(Tensor &other) {
    impl = impl->add(other.impl);
    return *this;
  }

  Tensor &operator*=(Tensor &other) {
    impl = impl->mul(other.impl);
    return *this;
  }

  Tensor &operator*=(float other) {
    impl = impl->mul(other);
    return *this;
  }

  Tensor &operator-=(Tensor &other) {
    impl = impl->sub(other.impl);
    return *this;
  }

  Tensor &operator/=(Tensor &other) {
    impl = impl->div(other.impl);
    return *this;
  }

  Tensor &reshape(std::vector<int> shape) {
    // -1 for inferring size
    int size = 1;
    int index = -1;
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] == -1) {
        if (index != -1) {
          throw std::runtime_error("only one dimension can be inferred");
        }
        index = i;
      } else {
        size *= shape[i];
      }
    }
    if (index != -1) {
      shape[index] = impl->size / size;
    }
    impl->shape = shape;
    return *this;
  }

  static Tensor from_kaiming_uniform(std::vector<int> shape) {
    // https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    size_t size = 1;
    for (auto &item : shape) {
      size *= item;
    }
    float bound = std::sqrt(6.0f / (static_cast<float>(size) / shape[0]));
    std::vector<float> data;
    for (size_t i = 0; i < size; i++) {
      float r = static_cast<float>(rand()) / RAND_MAX;
      data.push_back((r * 2.0f - 1.0f) * bound);
    }
    return Tensor(data, shape);
  }

  friend nn::SGD;
};

#endif // TENSORLIBRARY_TENSOR_H
