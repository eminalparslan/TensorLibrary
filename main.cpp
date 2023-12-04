#include "nn.h"
#include "optimizer.h"
#include "tensor.h"
#include <fstream>
#include <iostream>
#include <cassert>

void reverse_bytes(char *bytes, int size) {
  for (int i = 0; i < size / 2; i++) {
    std::swap(bytes[i], bytes[size - i - 1]);
  }
}

void load_MNIST(std::vector<std::pair<int, std::vector<float>>> &images) {
  std::ifstream inputf("./MNIST_ORG/train-images.idx3-ubyte");
  std::ifstream labelf("./MNIST_ORG/train-labels.idx1-ubyte");

  // TODO: print this out
  inputf.seekg(8);
  int x1, x2;
  inputf.read((char *)&x1, 4);
  inputf.read((char *)&x2, 4);
  reverse_bytes((char *)&x1, 4);
  reverse_bytes((char *)&x2, 4);

  int y1, y2;
  labelf.read((char *)&y1, 4);
  labelf.read((char *)&y2, 4);
  reverse_bytes((char *)&y1, 4);
  reverse_bytes((char *)&y2, 4);

  for (int i = 0; i < 60000; i++) {
    unsigned char c;
    labelf.read((char *)&c, 1);
    std::vector<float> image;
    for (int j = 0; j < x1 * x2; j++) {
      unsigned char b;
      inputf.read((char *)&b, 1);
      image.push_back((int)b);
    }
    images.push_back(std::make_pair((int)c, image));
  }

  inputf.close();
  labelf.close();
}

int main2() {
  std::vector<std::pair<int, std::vector<float>>> images;
  load_MNIST(images);

  nn::Linear l1 = nn::Linear(28 * 28, 128);
  nn::Linear l2 = nn::Linear(128, 64);
  nn::Linear l3 = nn::Linear(64, 10);

  nn::SGD sgd({l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias}, 0.001f);
  auto loss = nn::CrossEntropyLoss();

  int steps = 30000;
  assert(steps < (int)(images.size()));

  for (int i = 0; i < steps; i++) {
    auto image = images[i];
    int label = image.first;
    std::vector<float> data = image.second;

    // normalize
    for (auto &item : data) {
      item = item / 255;
    }

    sgd.zero_grad();

    Tensor a = Tensor(data, {1, 28 * 28});
    Tensor b = l1(a);
    Tensor c = b.relu();
    Tensor d = l2(c);
    Tensor e = d.relu();
    Tensor f = l3(e);
    Tensor h = loss(f.reshape({1, -1}), Tensor({static_cast<float>(label)}, {1}));
    // realize the computation graph
    h();
    // compute the gradients
    h.backward();
    // update the parameters
    sgd.step();
  }

  int test_steps = 1000;
  int correct = 0;

  for (int i = 0; i < test_steps; i++) {
    auto image = images[i + steps];
    int label = image.first;
    std::vector<float> data = image.second;

    // normalize
    for (auto &item : data) {
      item = item / 255;
    }

    Tensor a = Tensor(data, {1, 28 * 28});
    Tensor b = l1(a);
    Tensor c = b.relu();
    Tensor d = l2(c);
    Tensor e = d.relu();
    Tensor f = l3(e);
    f();

    if ((int) f.argmax()().item() == label) {
      correct++;
    }
  }

  printf("accuracy: %f\n", (float) correct / test_steps);

  return 0;
}

int main() {
  Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}).toDevice(Backend::CUDA);
  Tensor b = Tensor({-3.0f, 8.0f, 2.0f, 1.0f}, {1, 4}).toDevice(Backend::CUDA);
  Tensor c = Tensor({-8.0f, 2.0f, -3.0f, 7.0f}, {4}).toDevice(Backend::CUDA);
  Tensor e = a / b;
  Tensor d = -e + c;
  Tensor f = Tensor({-2.0f, 9.0f, 3.0f, 5.0f}, {1, 4}).toDevice(Backend::CUDA);
  Tensor g = d.relu() + f + e;
  Tensor L = g.cross_entropy_loss(Tensor({1.0f}, {1}).toDevice(Backend::CUDA));
  L();
  L.print();
  L.backward();
  
  // TODO: CUDA for matmul, CUDA grads, train on CUDA, maybe try lazy synchronization, batch performance, CUDA graph

  a.print_grad();
  b.print_grad();
  c.print_grad();
  
  return 0;
}
