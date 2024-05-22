#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix_rect.hpp"

TEST(testMatrix2D, sampleTest) {
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

TEST(testMatrix2D, nobroadcast) {
  Matrix2D<float> A(3, 4), B(3, 4);
  A(0, 0) = 1;
  B(0, 0) = 2;
  Matrix2D<float> C = A.dot(B);
  EXPECT_EQ(C.size_rows(), 3);
  EXPECT_EQ(C.size_cols(), 4);
  EXPECT_EQ(C(0, 0), 2);
  for (int i = 0; i < C.size_rows(); i++) {
    for (int j = 0; j < C.size_cols(); j++) {
      if (!(i == 0 && j == 0)) {
        EXPECT_EQ(C(i, j), 0);
      }
    }
  }
}

TEST(testMatrix2D, broadcast1) {
  Matrix2D<float> A(3, 4), B(1, 4);
  A(0, 0) = 1;
  A(1, 0) = 2;
  A(2, 0) = 3;
  B(0, 0) = 2;
  Matrix2D<float> C = A.dot(B);
  EXPECT_EQ(C.size_rows(), 3);
  EXPECT_EQ(C.size_cols(), 4);
  EXPECT_EQ(C(0, 0), 2);
  EXPECT_EQ(C(1, 0), 4);
  EXPECT_EQ(C(2, 0), 6);
  for (int i = 0; i < C.size_rows(); i++) {
    for (int j = 0; j < C.size_cols(); j++) {
      if (j != 0) {
        EXPECT_EQ(C(i, j), 0);
      }
    }
  }
}

TEST(testMatrix2D, broadcast2) {
  Matrix2D<float> A(3, 1), B(3, 4);
  A(0, 0) = 1;
  B(0, 0) = 2;
  B(0, 1) = 3;
  B(0, 2) = 4;
  B(0, 3) = 5;
  Matrix2D<float> C = A.dot(B);
  EXPECT_EQ(C.size_rows(), 3);
  EXPECT_EQ(C.size_cols(), 4);
  EXPECT_EQ(C(0, 0), 2);
  EXPECT_EQ(C(0, 1), 3);
  EXPECT_EQ(C(0, 2), 4);
  EXPECT_EQ(C(0, 3), 5);
  for (int i = 0; i < C.size_rows(); i++) {
    for (int j = 0; j < C.size_cols(); j++) {
      if (i != 0) {
        EXPECT_EQ(C(i, j), 0);
      }
    }
  }
}

TEST(testMatrix2D, broadcast3) {
  Matrix2D<float> A(3, 1), B(1, 4);
  A(0, 0) = 1;
  A(1, 0) = 6;
  A(2, 0) = 7;
  B(0, 0) = 2;
  B(0, 1) = 3;
  B(0, 2) = 4;
  B(0, 3) = 5;
  Matrix2D<float> C = A.dot(B);
  EXPECT_EQ(C.size_rows(), 3);
  EXPECT_EQ(C.size_cols(), 4);
  EXPECT_EQ(C(0, 0), 2);
  EXPECT_EQ(C(0, 1), 3);
  EXPECT_EQ(C(0, 2), 4);
  EXPECT_EQ(C(0, 3), 5);
  EXPECT_EQ(C(1, 0), 12);
  EXPECT_EQ(C(1, 1), 18);
  EXPECT_EQ(C(1, 2), 24);
  EXPECT_EQ(C(1, 3), 30);
  EXPECT_EQ(C(2, 0), 14);
  EXPECT_EQ(C(2, 1), 21);
  EXPECT_EQ(C(2, 2), 28);
  EXPECT_EQ(C(2, 3), 35);
}

TEST(testMatrix2D, broadcast4) {
  Matrix2D<float> A(1, 1), B(3, 4);
  A(0, 0) = 2;
  B(0, 0) = 2;
  B(0, 1) = 3;
  B(1, 2) = 4;
  B(2, 3) = 5;
  Matrix2D<float> C = A.dot(B);
  EXPECT_EQ(C.size_rows(), 3);
  EXPECT_EQ(C.size_cols(), 4);
  for (int i = 0; i < C.size_rows(); i++) {
    for (int j = 0; j < C.size_cols(); j++) {
      EXPECT_EQ(C(i, j), A(0, 0) * B(i, j));
    }
  }
}
