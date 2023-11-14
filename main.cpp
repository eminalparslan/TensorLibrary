#include "tensor.h"
#include <iostream>

int main() {
  Tensor a = Tensor({1, 2, 3, 4}, {2, 2});
  Tensor b = Tensor({-3, 8, 2, 1}, {2, 2});
  Tensor c = Tensor({10, -3, -8, 2}, {2, 2});
  Tensor e = a.matmul(b);
  Tensor d = e + c;
  Tensor f = Tensor({-2, 9, 3, 5}, {2, 2});
  // Tensor L = d.relu() + f;
  Tensor L = d.matmul(f);
  std::cout << "[";
  for (size_t i = 0; i < L.size; i++) {
    std::cout << L.data[i] << ",";
  }
  std::cout << "]" << std::endl;

  // set L.grad to 1
  for (size_t i = 0; i < L.size; i++) {
    L.grad[i] = 1;
  }
  L.backward();
  std::cout << "[";
  for (size_t i = 0; i < a.size; i++) {
    std::cout << a.grad[i] << ",";
  }
  std::cout << "]" << std::endl;
  return 0;
}
