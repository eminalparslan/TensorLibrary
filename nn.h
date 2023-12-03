//
// Created by Emin Arslan on 11/9/23.
//

#ifndef TENSORLIBRARY_NN_H
#define TENSORLIBRARY_NN_H

#include "tensor.h"

class Layer {
public:
  virtual Tensor forward(Tensor x) = 0;
  Tensor operator()(Tensor x) { return forward(x); }
  virtual Tensor backward() = 0;
};

namespace nn {

class Linear {
public:
  Tensor weight;
  Tensor bias;

  Linear(int in_features, int out_features)
      : weight(Tensor::from_kaiming_uniform({in_features, out_features})),
        bias(Tensor::from_kaiming_uniform({out_features})) {}

  Tensor operator()(Tensor x) { return x.matmul(weight) + bias; }
};

class ReLU {
public:
  Tensor operator()(Tensor x) { return x.relu(); }
};

class CrossEntropyLoss {
public:
  Tensor operator()(Tensor x, Tensor y) { return x.cross_entropy_loss(y); }
};

} // namespace nn

#endif // TENSORLIBRARY_NN_H
