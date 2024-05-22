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
  if (!(A.size_rows() == B.size_rows() || A.size_rows() == 1 ||
        B.size_rows() == 1)) {
    return false;
  }
  if (!(A.size_cols() == B.size_cols() || A.size_cols() == 1 ||
        B.size_cols() == 1)) {
    return false;
  }
  return true;
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
  Matrix2D() : n_rows(0), n_cols(0), data_(nullptr) {}

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) : n_rows(m), n_cols(n) {
    // TODO: Hint: The data_ should be resized to have m * n elements
    if (n < 0 || m < 0) {
      throw std::runtime_error(
          "Negative matrix dimension passed to constructor.");
    }
    data_ = new T[m * n];
    std::fill_n(data_, m * n, static_cast<T>(0));
  }

  // Destructor
  ~Matrix2D() {
      delete[] data_;
  }

  // Copy constructor
  Matrix2D(const Matrix2D& other) : n_rows(other.n_rows), n_cols(other.n_cols) {
    data_ = new T[n_rows * n_cols];
    std::copy(other.data_, other.data_ + n_rows * n_cols, data_);
  }

  // Copy assignment operator
  Matrix2D& operator=(const Matrix2D& other) {
    if (this != &other) {
      delete[] data_;
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
    other.n_rows = 0;
    other.n_cols = 0;
    other.data_ = nullptr;
  }

  // Move assignment operator
  Matrix2D& operator=(Matrix2D&& other) noexcept {
    if (this != &other) {
      delete[] data_;
      n_rows = other.n_rows;
      n_cols = other.n_cols;
      data_ = other.data_;
      other.n_rows = 0;
      other.n_cols = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  unsigned int size_rows() const { return n_rows; } // TODO
  unsigned int size_cols() const { return n_cols; } // TODO

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    // TODO: Hint: Element (i,j) for 0 <= i < n_rows and 0 <= j < n_cols 
    // is stored at data[i * n_cols + j]. 
    if (i < 0 || i >= n_rows || j < 0 || j >= n_cols)
      throw std::out_of_range("Index out of range.");
    return data_[i * n_cols + j];
  }
    
  void Print(std::ostream& ostream) {
      // TODO
    for (std::size_t i = 0; i < n_rows; i++) {
      for (std::size_t j = 0; j < n_cols; j++)
        ostream << std::setw(5) << (*this)(i, j) << " ";
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
      int result_rows = n_rows > mat.size_rows() ? n_rows : mat.size_rows();
      int result_cols = n_cols > mat.size_cols() ? n_cols : mat.size_cols();
      Matrix2D<T> ret(result_rows, result_cols);

      for (int i = 0; i < result_rows; i++) {
        int row_idx = n_rows == 1 ? 0 : i;
        int mat_row_idx = mat.size_rows() == 1 ? 0 : i;

        for (int j = 0; j < result_cols; j++) {
          int col_idx = n_cols == 1 ? 0 : j;
          int mat_col_idx = mat.size_cols() == 1 ? 0 : j;

          ret(i, j) = (*this)(row_idx, col_idx) * mat(mat_row_idx, mat_col_idx);
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
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
