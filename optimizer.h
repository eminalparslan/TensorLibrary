//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_OPTIMIZER_H
#define TENSORLIBRARY_OPTIMIZER_H

#include "tensor.h"

namespace nn {

class Optimizer {
public:
  std::vector<Tensor> parameters;
  float learning_rate;

  Optimizer(std::vector<Tensor> parameters, float learning_rate)
      : parameters(parameters), learning_rate(learning_rate) {}

  virtual void step() = 0;
  virtual void zero_grad() {
    for (auto &param : this->parameters) {
      param.zero_grad();
    }
  }
};

class SGD : public Optimizer {
public:
  float momentum;

  SGD(std::vector<Tensor> parameters, float learning_rate,
      float momentum = 0.0f)
      : Optimizer(parameters, learning_rate), momentum(momentum) {}

  void step() {
    for (auto &param : this->parameters) {
      for (size_t i = 0; i < param.size(); i++) {
        param.impl->data[i] -= param.impl->grad[i] * this->learning_rate;
      }
      /* if (this->momentum > 0.0f) { */
      /*   Tensor v = param.grad() * this->momentum; */
      /*   param -= v; */
      /* } */
    }
  }
};

} // namespace nn

#endif // TENSORLIBRARY_OPTIMIZER_H
