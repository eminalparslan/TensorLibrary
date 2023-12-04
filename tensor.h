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
#include <cuda_runtime.h>

#include "kernels.h"

namespace nn {
class SGD;
}

enum class Backend { CPU, CUDA };


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

  float *cuda_data{nullptr};
  float *cuda_grad{nullptr};

  std::vector<int> shape;
  size_t size;

  // TODO: stream per Tensor, lazily synchronize on streams when needed
  Backend backend{Backend::CPU};
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
             std::initializer_list<TensorImpl *> children, Backend backend)
      : children(children), shape(shape), backend(backend) {
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
    if (backend == Backend::CUDA) {
      cudaMalloc(&cuda_data, this->size * sizeof(float));
      cudaMalloc(&cuda_grad, this->size * sizeof(float));
      cudaMemset(cuda_data, 0, this->size * sizeof(float));
      cudaMemset(cuda_grad, 0, this->size * sizeof(float));
    }
  }

  ~TensorImpl() {
    delete[] data;
    delete[] grad;
    if (backend == Backend::CUDA) {
      cudaFree(cuda_data);
      cudaFree(cuda_grad);
    }
  }
  
  // TODO: consider carrying gradients through copy constructor
  TensorImpl(const TensorImpl &other) = delete;
  
  void toDevice(Backend backend) {
    if (this->backend == backend) {
      return;
    } else if (this->backend == Backend::CUDA) {
      if (backend == Backend::CPU) {
        cudaMemcpy(data, cuda_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(grad, cuda_grad, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(cuda_data);
        //cudaFree(cuda_grad);
      }
      this->backend = Backend::CPU;
    } else if (this->backend == Backend::CPU) {
      if (backend == Backend::CUDA) {
        cudaMalloc(&cuda_data, size * sizeof(float));
        //cudaMalloc(&cuda_grad, size * sizeof(float));
        cudaMemcpy(cuda_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(cuda_grad, grad, size * sizeof(float), cudaMemcpyHostToDevice);
      }
      this->backend = Backend::CUDA;
    }
  }

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
    assert(realized);

    for (size_t i = 0; i < size; i++) {
      grad[i] = 1.0f;
    }
    /*if (backend == Backend::CUDA) {*/
    /*  cudaMemcpy(cuda_grad, grad, size * sizeof(float), cudaMemcpyHostToDevice);*/
    /*}*/

    std::vector<TensorImpl *> toposort;
    std::unordered_set<TensorImpl *> visited;
    topo_sort(visited, toposort);

    for (auto it = toposort.rbegin(); it != toposort.rend(); ++it) {
      // TODO: remove
      enum Backend prev = (*it)->backend;
      (*it)->backend = Backend::CPU;

      if ((*it)->grad_fn != nullptr) (*it)->grad_fn();
      if ((*it)->backend == Backend::CUDA) cudaDeviceSynchronize();

      (*it)->backend = prev;
    }
    
    /*for (auto &item : toposort) {*/
    /*  if (item->backend == Backend::CUDA) {*/
    /*    cudaMemcpy(item->grad, item->cuda_grad, item->size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*    item->backend = Backend::CPU;*/
    /*  }*/
    /*}*/
  }

  void print() {
    // TODO: check if realized
    std::cout << "[";
    for (size_t i = 0; i < this->size; i++) {
      std::cout << this->data[i] << ",";
    }
    std::cout << "]" << std::endl;
  }
  
  void print_grad() {
    std::cout << "[";
    for (size_t i = 0; i < this->size; i++) {
      std::cout << this->grad[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  void forward() {
    std::vector<TensorImpl *> toposort;
    std::unordered_set<TensorImpl *> visited;
    topo_sort(visited, toposort);

    for (auto &item : toposort) {
      if (item->realized) continue;
      if (item->calc_fn != nullptr) item->calc_fn();
      if (item->backend == Backend::CUDA) cudaDeviceSynchronize();
    }
    
    for (auto &item : toposort) {
      if (item->backend == Backend::CUDA) {
        cudaMemcpy(item->data, item->cuda_data, item->size * sizeof(float), cudaMemcpyDeviceToHost);
      }
      item->realized = true;
    }
  }

  void zero_grad() {
    /*if (backend == Backend::CPU) {*/
      for (size_t i = 0; i < size; i++) {
        grad[i] = 0.0f;
      }
    /*} else if (backend == Backend::CUDA) {*/
    /*  cudaMemset(cuda_grad, 0, size * sizeof(float));*/
    /*}*/
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
  
  Tensor &toDevice(Backend backend) {
    impl->toDevice(backend);
    return *this;
  }

  float item() {
    assert(impl->backend == Backend::CPU);
    if (impl->size != 1) {
      throw std::runtime_error("item() only supported for size 1 Tensors");
    }
    return impl->data[0];
  }

  void print() {
    // TODO: check if realized
    std::cout << "[";
    for (size_t i = 0; i < impl->size; i++) {
      std::cout << impl->data[i] << ",";
    }
    std::cout << "]" << std::endl;
  }

  void print_grad() {
    // TODO: check if grads are realized
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
    // FIXME: probably incorrect, matmul example isn't correct
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
