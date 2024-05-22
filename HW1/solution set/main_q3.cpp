#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Matrix {
 public:
  virtual ~Matrix() = 0;
  virtual std::string repr() = 0;
};

Matrix::~Matrix() {}

class SparseMatrix : public Matrix {
 public:
  std::string repr() { return "sparse"; }
};

class ToeplitzMatrix : public Matrix {
 public:
  std::string repr() { return "toeplitz"; }
};

void PrintRepr(const std::vector<std::shared_ptr<Matrix>> &vec) {
  for (auto &v : vec) std::cout << v->repr() << std::endl;
}

int main() {
  std::vector<std::shared_ptr<Matrix>> vec;
  vec.push_back(std::make_shared<SparseMatrix>());
  vec.push_back(std::make_shared<ToeplitzMatrix>());
  PrintRepr(vec);
}
