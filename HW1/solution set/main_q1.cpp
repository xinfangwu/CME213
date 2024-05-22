#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, sampleTest) {
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

TEST(testMatrix, diagonalTest1) {
  // Empty matrix
  MatrixDiagonal<float> matrix;
  EXPECT_TRUE(matrix.NormL0() == 0) << "Incorrect default constructor.";
  EXPECT_TRUE(matrix.size() == 0) << "Incorrect default constructor.";
}

TEST(testMatrix, diagonalTest2) {
  // Matrix of size 0
  MatrixDiagonal<float> matrix2(0);
  EXPECT_TRUE(matrix2.NormL0() == 0) << "Empty Matrix incorrect l0 norm.";
  EXPECT_TRUE(matrix2.size() == 0) << "Empty Matrix incorrect l0 norm.";
}

TEST(testMatrix, diagonalTest3) {
  // Negative size
  EXPECT_ANY_THROW(MatrixDiagonal<float> matrix(-1))
      << "Error: negative size constructor.";
}

TEST(testMatrix, diagonalTest4) {
  // Initialize and test L0 norm
  MatrixDiagonal<double> matrix3(100);
  EXPECT_EQ(matrix3.NormL0(), 0);
  EXPECT_EQ(matrix3.size(), 100);
  matrix3.set(1, 1, 4);
  matrix3.set(2, 2, 1);
  matrix3.set(10, 10, -3);
  EXPECT_EQ(matrix3.NormL0(), 3);
}

TEST(testMatrix, diagonalTest5) {
  // Initializing and retrieving values
  int n = 10;
  MatrixDiagonal<long> matrix4(n);

  for (int i = 0; i < n; i++) matrix4.set(i, i, i);

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(matrix4(i, i), i);
    for (int j = 0; j < n; j++) {
      if (j != i) {
        EXPECT_EQ(matrix4(i, j), 0);
        EXPECT_EQ(matrix4(j, i), 0);
      }
    }
  }
}

TEST(testMatrix, diagonalTest6) {
  // Out of bounds
  MatrixDiagonal<short> matrix5(10);
  EXPECT_ANY_THROW(matrix5(10, 0));

  MatrixDiagonal<short> matrix6(4);
  EXPECT_ANY_THROW(matrix6(0, 4));

  MatrixDiagonal<short> matrix7(3);
  EXPECT_ANY_THROW(matrix7(-1, 0));
}

TEST(testMatrix, diagonalTest7) {
  // Test stream operator
  MatrixSymmetric<int> matrix8(2);
  std::stringstream ss("");
  matrix8.set(0, 0, 1);
  matrix8.set(1, 1, 2);
  ss << matrix8;

  EXPECT_EQ(ss.str(), "    1     0 \n    0     2 \n");
}

TEST(testMatrix, symmetricTest1) {
  // Empty matrix
  MatrixSymmetric<float> matrix;
  EXPECT_TRUE(matrix.NormL0() == 0) << "Incorrect default constructor.";
  EXPECT_TRUE(matrix.size() == 0) << "Incorrect default constructor.";
}

TEST(testMatrix, symmetricTest2) {
  // Matrix of size 0
  MatrixSymmetric<float> matrix2(0);
  EXPECT_TRUE(matrix2.NormL0() == 0) << "Empty Matrix incorrect l0 norm.";
  EXPECT_TRUE(matrix2.size() == 0) << "Empty Matrix incorrect l0 norm.";
}

TEST(testMatrix, symmetricTest3) {
  // Negative size
  EXPECT_ANY_THROW(MatrixSymmetric<float> matrix(-1))
      << "Error: negative size constructor.";
}

TEST(testMatrix, symmetricTest4) {
  // Initialize and test L0 norm
  MatrixSymmetric<double> matrix3(100);
  EXPECT_EQ(matrix3.NormL0(), 0);
  EXPECT_EQ(matrix3.size(), 100);
  matrix3.set(1, 1, 4);
  matrix3.set(2, 3, 1);
  matrix3.set(10, 1, -3);
  EXPECT_EQ(matrix3.NormL0(), 5);
}

TEST(testMatrix, symmetricTest5) {
  // Initializing and retrieving values
  int n = 10;
  MatrixSymmetric<long> matrix4(n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) matrix4.set(i, j, n * i + j);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      EXPECT_EQ(matrix4(i, j), n * i + j);
      EXPECT_EQ(matrix4(j, i), n * i + j);
    }
  }
}

TEST(testMatrix, symmetricTest6) {
  // Out of bounds
  MatrixSymmetric<short> matrix5(10);
  EXPECT_ANY_THROW(matrix5(10, 0));

  MatrixSymmetric<short> matrix6(4);
  EXPECT_ANY_THROW(matrix6(0, 4));

  MatrixSymmetric<short> matrix7(3);
  EXPECT_ANY_THROW(matrix7(-1, 0));
}

TEST(testMatrix, symmetricTest7) {
  // Test stream operator
  MatrixSymmetric<int> matrix8(2);
  std::stringstream ss("");
  matrix8.set(0, 0, 1);
  matrix8.set(0, 1, 2);
  matrix8.set(1, 1, 3);
  ss << matrix8;

  EXPECT_EQ(ss.str(), "    1     2 \n    2     3 \n");
}
