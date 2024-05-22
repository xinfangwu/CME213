#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  public:
  // Sets value of matrix element (i, j).
  virtual void set(int i, int j, T value) = 0;
  // Returns value of matrix element (i, j).
  virtual T operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  m.Print(stream);
  return stream;
}

/* MatrixDiagonal Class is a subclass of the Matrix class */
template <typename T>
class MatrixDiagonal : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  // store diagonals only
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixDiagonal() : n_(0) {}

  // Constructor takes argument n = matrix dimension.
  MatrixDiagonal(const int n) : n_(n) {
    if (n < 0) {
      throw std::runtime_error(
          "Negative matrix dimension passed to constructor.");
    }
    data_.resize(n, static_cast<T>(0));
  }

  unsigned int size() const { return n_; }

  // Sets value of matrix element (i, j).
  void set(int i, int j, T value) override {
    if (i < 0 || i >= n_ || j < 0 || j >= n_)
      throw std::out_of_range("Index out of range.");
    if (j == i) data_[j] = value;
    else {
      throw std::runtime_error("Cannot set value of off-diagonal element.");
    };
  }

  // Returns value of matrix element (i, j).
  T operator()(int i, int j) override {
    if (i < 0 || i >= n_ || j < 0 || j >= n_)
      throw std::out_of_range("Index out of range.");
    if (j == i) return data_[j];
    else {
      return static_cast<T>(0);
    };
  }

  // Number of non-zero elements in matrix.
  unsigned NormL0() const override {
    if (n_ == 0) return 0;

    unsigned int all_nnz =
        std::count_if(data_.begin(), data_.end(),
                      [](const T& val) { return val != static_cast<T>(0); });

    return all_nnz;
  }

  void Print(std::ostream& ostream) override {
    for (std::size_t i = 0; i < n_; i++) {
      for (std::size_t j = 0; j < n_; j++)
        ostream << std::setw(5) << (*this)(i, j) << " ";
      ostream << std::endl;
    }
  }
};

template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  unsigned int n_;

  // store lower triangular matrix only
  std::vector<T> data_;

 public:
  // Empty matrix
  MatrixSymmetric() : n_(0) { data_.clear(); }

  // Constructor takes argument n = matrix dimension.
  MatrixSymmetric(const int n) : n_(n) {
    if (n < 0) {
      throw std::runtime_error(
          "Negative matrix dimension passed to constructor.");
    }
    data_.resize(n * (n + 1) / 2, static_cast<T>(0));
  }

  unsigned int size() const { return n_; }

  // Sets value of matrix element (i, j).
  void set(int i, int j, T value) override {
    if (i < 0 || i >= n_ || j < 0 || j >= n_)
      throw std::out_of_range("Index out of range.");
    if (j > i) std::swap(i, j);
    data_[j + i * (i + 1) / 2] = value;
  }

  // Returns value of matrix element (i, j).
  T operator()(int i, int j) override {
    if (i < 0 || i >= n_ || j < 0 || j >= n_)
      throw std::out_of_range("Index out of range.");
    if (j > i) std::swap(i, j);
    return data_[j + i * (i + 1) / 2];
  }

  // Number of non-zero elements in matrix.
  unsigned NormL0() const override {
    if (n_ == 0) return 0;

    // When computing the non-zero elements in a matrix, we need to
    // differentiate between diagonal and non-diagonal entries. We need to count
    // diagonal entries once and off-diagonal entries twice. Since we only store
    // the lower triangular part of the matrix, the ith entry along the diagonal
    // is stored at index (i+1)*(i+2) / 2 - 1 of data_.
    unsigned int all_nnz = 2 * std::count_if(data_.begin(), data_.end(),
        [] (const T& val) { return val != static_cast<T>(0); }
    );
    for (int i = 0; i < n_; i++) {
      if (data_[(i+1)*(i+2)/2 - 1] != static_cast<T>(0)) {
        all_nnz -= 1;
      }
    }
    return all_nnz;
  }

  void Print(std::ostream& ostream) override {
    for (std::size_t i = 0; i < n_; i++) {
      for (std::size_t j = 0; j < n_; j++)
        ostream << std::setw(5) << (*this)(i, j) << " ";
      ostream << std::endl;
    }
  }
};

#endif /* MATRIX_HPP */
