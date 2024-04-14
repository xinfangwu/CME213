#ifndef MATRIX_RECT
#define MATRIX_RECT

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

template <typename T>
class Matrix2D;

template <typename T>
bool Broadcastable(Matrix2D<T>& A, Matrix2D<T>& B) {
  // TODO: Write a function that returns true if either of the matrices can be
  // broadcast to be compatible with the other for elementwise multiplication.

  // first check if the matrix is empty 
  if((A.size_cols() ==  0 && A.size_rows() == 0) || (B.size_cols() == 0 && B.size_rows() == 0)){
    return false;
    // throw std::invalid_argument("empty matrix is not broadcastable");
  }
  // they are equal, or one of them is 1.
  if((A.size_cols() == B.size_cols() || A.size_cols() == 1 || B.size_cols() == 1) &&
    (A.size_rows() == B.size_rows() || A.size_rows() == 1 || B.size_rows() == 1) ){
    return true;
  }
  else{
    return false;
  }
}

template <typename T>
class Matrix2D {
 private:
  // The size of the matrix is (n_rows, n_cols)
  unsigned int n_rows;
  unsigned int n_cols;

  // Dynamic array storing the data in row major order. Element (i,j) for 
  // 0 <= i < n_rows and 0 <= j < n_cols is stored at data[i * n_cols + j].
  T* data_;

 public:
  // Empty matrix
  Matrix2D() : n_rows(0), n_cols(0), data_(nullptr) { 
    // TODO
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) {
      // TODO: Hint: allocate memory for m * n elements using keyword 'new'
      n_rows = m;
      n_cols = n;
      data_ = new T[m*n];
  }

  // Destructor
  ~Matrix2D() {
    // TODO: Hint: Use keyword 'delete'
    delete [] data_;
  }

  // Copy constructor
  Matrix2D(const Matrix2D& other) : n_rows(other.n_rows), n_cols(other.n_cols) {
    // TODO
    data_ = new T[n_rows * n_cols];
    std::copy(other.data_, other.data_ + n_rows * n_cols, data_);
  }

  // Copy assignment operator
  Matrix2D& operator=(const Matrix2D& other) {
    // TODO
    // return *this;
    if (this != &other){
      delete [] data_;
      n_rows = other.n_rows;
      n_cols = other.n_cols;
      data_ = new T[n_rows * n_cols];
      std::copy(other.data_, other.data_ + n_rows * n_cols, data_);
    }
    return *this;
  }

  // Move constructor
  Matrix2D(Matrix2D&& other) noexcept 
    : n_rows(other.n_rows), n_cols(other.n_cols), data_(other.data_) {
    // TODO
    other.data_ = nullptr;
    other.n_rows = 0;
    other.n_cols = 0;
  }

  // Move assignment operator
  Matrix2D& operator=(Matrix2D&& other) noexcept {
    // TODO
    // return *this;
    if (this != &other){
      delete [] data_;
      n_rows = other.n_rows;
      n_cols = other.n_cols;
      data_ = other.data_;
      other.data_ = nullptr;
      other.n_rows = 0;
      other.n_cols = 0;
    }
    return *this;
  }

  unsigned int size_rows() const { return n_rows; } // TODO
  unsigned int size_cols() const { return n_cols; } // TODO

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    // TODO: Hint: Element (i,j) for 0 <= i < n_rows and 0 <= j < n_cols 
    // is stored at data[i * n_cols + j]. 
    // return data_[0];
    if(i < n_rows && j <n_cols && i >= 0 && j >= 0){
      return data_[i * n_cols + j];
    }
    else{
      throw std::out_of_range("Index out of range");
    }
  }
    
  void Print(std::ostream& ostream) {
    // TODO
    for (int i=0; i<n_rows; i++){
      for (int j=0; j<n_cols; j++){
        ostream << operator()(i, j) << " ";
      }
      ostream << std::endl;
    }
  }

  Matrix2D<T> dot(Matrix2D<T>& mat) {
    if (n_rows == mat.size_rows() && n_cols == mat.size_cols()) {
      Matrix2D<T> ret(n_rows, n_cols);
      for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
          ret(i, j) = (*this)(i, j) * mat(i, j);
        }
      }
      return ret;
    } else if (Broadcastable<T>(*this, mat)) {
      // TODO: Replace the code in this scope.
      // Compute and return the elementwise product of the two Matrix2D's
      // "*this" and "mat" after appropriate broadcasting.
      int retrows = std::max(n_rows, mat.size_rows());
      int retcols = std::max(n_cols, mat.size_cols());
      
      Matrix2D<T> ret(retrows, retcols);

      // row 
      for(int i=0; i<retrows; i++){
        for(int j=0; j<retcols; j++){
          T val1, val2;
          // this matrix 
          if(i<n_rows && j<n_cols){
            val1 = (*this)(i, j);
          } else{
            if(i >= n_rows && j<n_cols){
              val1 = (*this)(0, j);
            } else if (i < n_rows && j >= n_cols){
              val1 = (*this)(i, 0);
            } else {
              val1 = (*this)(0, 0);
            }
          }
          // mat matrix 
          if(i<mat.size_rows() && j < mat.size_cols()){
            val2 = mat(i, j);
          } else{
            if(i >= mat.size_rows() && j< mat.size_cols()){
              val2 = mat(0, j);
            } else if (i < mat.size_rows() && j >= mat.size_cols()){
              val2 = mat(i, 0);
            } else {
              val2 = mat(0, 0);
            }
          }

          ret(i, j) = val1 * val2;
        }
      }

      return ret;
    } else {
      throw std::invalid_argument("Incompatible shapes of the two matrices.");
    }
  }

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix2D<U>& m);
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix2D<T>& m) {
  // V TODO
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
