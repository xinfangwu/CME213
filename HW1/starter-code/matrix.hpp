#ifndef MATRIX_HPP
#define MATRIX_HPP

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
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixDiagonal() : n_(0), data_() {}

  // TODO: Constructor that takes matrix dimension as argument
  MatrixDiagonal(const int n) : n_(n), data_(n)
  {
    if (n < 0) throw std::length_error("Matrix dimension must be non-negative.");
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that sets value of matrix element (i, j).
  void set(int i, int j, T value) override 
  {  
    if (i < 0 || i >= n_ || j< 0 || j >= n_){
      throw std::out_of_range("Index out of bounds.");
    }
    else if(i != j){
      throw std::invalid_argument("Only diagonal elements can be set.");
    }
    else{
      data_[i] = value;
    }
  }

  // TODO: Function that returns value of matrix element (i, j).
  T operator()(int i, int j) override { 
    if (i < 0 || i >= n_ || j< 0 || j >= n_ ){
      throw std::out_of_range("Index out of bounds.");
    }
    else if(i != j){
      return 0;
    }
    else{
      return data_[i];
    }
   }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override 
  { 
    int count = 0;
    for (int i=0; i<n_; i++)
    {
      if(data_[i] != 0){
        count++;
      }
    }
    return count;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (int i=0; i<n_; i++){
      for (int j=0; j<n_; j++){
        ostream << operator()(i, j) << " ";
      }
      ostream <<std::endl;
    }
  }
};

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixSymmetric() : n_(0), data_() {}

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) : n_(n), data_(n*(n+1)/2)
  {
    if (n < 0) throw std::length_error("Matrix dimension must be non-negative.");
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that sets value of matrix element (i, j).
  void set(int i, int j, T value) override {
    if (i < 0 || i >= n_ || j <0 || j >= n_){
      throw std::out_of_range("Index out of bounds.");
    }
    else{
      if (i < j){
        std::swap(i, j);
      }
      data_[i*(i+1)/2 + j] = value;
   }
  }

  // TODO: Function that returns value of matrix element (i, j).
  T operator()(int i, int j) override { 
    if (i < 0 || i >= n_ || j <0 || j >= n_){
      throw std::out_of_range("Index out of bounds.");
    }
    else{
      if (i < j){
        std::swap(i, j);
      }
      return data_[i*(i+1)/2 + j];
    }
   }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override { 
    int count = 0;
    for (int j=0; j<n_; j++){
      for (int i=0; i<=j; i++){
        if (i==j){
          if (data_[i*(i+1)/2 + j] != 0){
            count += 1;
          }
        }
        else{
          if (data_[i*(i+1)/2 + j] != 0){
            count += 2;
          }
        }
      }
    }
    return count;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (int i=0; i<n_; i++){
      for (int j=0; j<n_; j++){
        ostream << operator()(i, j) << " ";
      }
      ostream <<std::endl;
    }
  }
};

#endif /* MATRIX_HPP */