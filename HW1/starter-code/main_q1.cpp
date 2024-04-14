#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

// TEST(testMatrix, sampleTest) {
//   ASSERT_EQ(1000, 1000)
//       << "This does not fail, hence this message is not printed.";
//   EXPECT_EQ(2000, 2000)
//       << "This does not fail, hence this message is not printed.";
//   // If uncommented, the following line will make this test fail.
//   // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
//   // will be printed.";
// }

// 1. Declare an empty matrix with the default constructor for MatrixSymmetric.
// Assert that the NormL0 and size functions return appropriate values for these.
TEST(testMatrix, defaultDiagonalMatrix) {
  MatrixDiagonal<int> defautMatrix;
  ASSERT_EQ(defautMatrix.NormL0(), 0) << "default diagonal matrix has wrong NormL0";
  ASSERT_EQ(defautMatrix.size(), 0) << "default diagonal matrix has wrong size";
}

TEST(testMatrix, defaultSymmetricMatrix) {
  MatrixSymmetric<int> defautMatrix;
  ASSERT_EQ(defautMatrix.NormL0(), 0) << "default symmetric matrix has wrong NormL0";
  ASSERT_EQ(defautMatrix.size(), 0) << "default symmetric matrix has wrong size";
}


// 2. Using the second constructor that takes size as argument, create a matrix of
// size zero. Repeat the assertions from (1).
TEST(testMatrix, zeroDiagonalMatrix) {
  MatrixDiagonal<int> zeroMatrix(0);
  ASSERT_EQ(zeroMatrix.NormL0(), 0) << "zero diagonal matrix has wrong NormL0";
  ASSERT_EQ(zeroMatrix.size(), 0) << "zero diagonal matrix has wrong size";
}

TEST(testMatrix, zeroSymmetricMatrix) {
  MatrixSymmetric<int> zeroMatrix(0);
  ASSERT_EQ(zeroMatrix.NormL0(), 0) << "zero symmetric matrix has wrong NormL0";
  ASSERT_EQ(zeroMatrix.size(), 0) << "zero symmetric matrix has wrong size";
}


// 3. Provide a negative argument to the second constructor and assert that the
// constructor throws an exception.
TEST(testMatrix, negativeDiagonalMatrix) {
  EXPECT_THROW(MatrixDiagonal<int> negativeMatrix(-7), std::length_error);
}

TEST(testMatrix, negativeSymmetricMatrix) {
  EXPECT_THROW(MatrixSymmetric<int> negativeMatrix(-10), std::length_error);
}

// 4. Create and initialize a matrix of some size, and verify that the NormL0
// function returns the correct value.
TEST(testMatrix, normalDiagonalMatrix) {
  MatrixDiagonal<int> dmatrix(5);
    ASSERT_EQ(dmatrix.size(), 5) << "normal diagonal size wrong";
    dmatrix.set(0, 0, 1);
    dmatrix.set(1, 1, 2);
    dmatrix.set(2, 2, 3);
    dmatrix.set(3, 3, 0);
    dmatrix.set(4, 4, 0);
    ASSERT_EQ(dmatrix.NormL0(), 3) << "normal diagonal NormL0 wrong";
    dmatrix.set(3, 3, 4);
    ASSERT_EQ(dmatrix.NormL0(), 4) << "normal diagonal NormL0 wrong";
}

TEST(testMatrix, normalSymmetricMatrix) {
  MatrixSymmetric<int> smatrix(3);
  ASSERT_EQ(smatrix.size(), 3) << "normal symmetric size wrong";
  smatrix.set(0, 0, 0);
  smatrix.set(0, 1, 2);
  smatrix.set(0, 2, 3);
  smatrix.set(1, 1, 4);
  smatrix.set(1, 2, 5);
  smatrix.set(2, 2, 6);
  ASSERT_EQ(smatrix.NormL0(), 8) << "normal symmetric NormL0 wrong";
  smatrix.set(2, 2, 0);
  ASSERT_EQ(smatrix.NormL0(), 7) << "normal symmetric NormL0 wrong";
  smatrix.set(0, 1, 0);
  ASSERT_EQ(smatrix.NormL0(), 5) << "normal symmetric NormL0 wrong";
}

// 5. Create a matrix, initialize some or all of its elements, then retrieve and
// check that they are what you initialized them to.
TEST(testMatrix, retrieveDiagonalMatrix) {
  MatrixDiagonal<int> dmatrix(5);
    ASSERT_EQ(dmatrix.size(), 5) << "retrieve diagonal size wrong";
    dmatrix.set(0, 0, 1);
    dmatrix.set(1, 1, 2);
    dmatrix.set(2, 2, 3);
    dmatrix.set(3, 3, 4);
    dmatrix.set(4, 4, 5);
    ASSERT_EQ(dmatrix(4, 4), 5) << "retrieve wrong value";
    ASSERT_EQ(dmatrix(2, 2), 3) << "retrieve wrong value";
    dmatrix.set(3, 3, 7);
    ASSERT_EQ(dmatrix(3, 3), 7) << "retrieve wrong value";
    EXPECT_THROW(dmatrix(7, 7),  std::out_of_range) << "out of range";
}

TEST(testMatrix, retrieveSymmetricMatrix) {
  MatrixSymmetric<int> smatrix(3);
  ASSERT_EQ(smatrix.size(), 3) << "retrieve symmetric size wrong";
  smatrix.set(0, 0, 0);
  smatrix.set(0, 1, 2);
  smatrix.set(0, 2, 3);
  smatrix.set(1, 1, 4);
  smatrix.set(1, 2, 5);
  smatrix.set(2, 2, 6);
  ASSERT_EQ(smatrix(0, 1), 2) << "retrieve wrong value";
  ASSERT_EQ(smatrix(1, 0), 2) << "retrieve wrong value";
  ASSERT_EQ(smatrix(2, 2), 6) << "retrieve wrong value";
  smatrix.set(2, 2, 0);
  ASSERT_EQ(smatrix(2, 2), 0) << "retrieve wrong value";
  EXPECT_THROW(smatrix(7, 7),  std::out_of_range) << "out of range";
}

// 6. Create a matrix of some size. Make an out-of-bounds access into it and check
// that an exception is thrown.
TEST(testMatrix, outDiagonalMatrix) {
  MatrixDiagonal<int> dmatrix(5);
    dmatrix.set(0, 0, 1);
    dmatrix.set(1, 1, 2);
    dmatrix.set(2, 2, 3);
    dmatrix.set(3, 3, 4);
    dmatrix.set(4, 4, 5);
    dmatrix.set(3, 3, 7);
    EXPECT_THROW(dmatrix(7, 7),  std::out_of_range) << "out of range";
    EXPECT_THROW(dmatrix(-1, 7),  std::out_of_range) << "out of range";

}

TEST(testMatrix, outSymmetricMatrix) {
  MatrixSymmetric<int> smatrix(3);
  smatrix.set(0, 0, 0);
  smatrix.set(0, 1, 2);
  smatrix.set(0, 2, 3);
  smatrix.set(1, 1, 4);
  smatrix.set(1, 2, 5);
  smatrix.set(2, 2, 6);
  EXPECT_THROW(smatrix(7, 7),  std::out_of_range) << "out of range";
  EXPECT_THROW(smatrix(-10, 7),  std::out_of_range) << "out of range";

}
// 7. Test the stream operator using std::stringstream and using the "<<" operator.
TEST(testMatrix, printDiagonalMatrix){
    MatrixDiagonal<int> dmatrix(3);
    dmatrix.set(0, 0, 1);
    dmatrix.set(1, 1, 2);
    dmatrix.set(2, 2, 3);

    std::stringstream ss;
    ss << dmatrix;
    ASSERT_EQ(ss.str(), "1 0 0 \n0 2 0 \n0 0 3 \n") << "wrong result" ;
}

TEST(testMatrix, printSymmetricMatrix){
    MatrixSymmetric<int> smatrix(3);
    std::stringstream ss;
    
    smatrix.set(0, 0, 1);
    smatrix.set(0, 1, 2);
    smatrix.set(0, 2, 3);
    smatrix.set(1, 1, 4);
    smatrix.set(1, 2, 5);
    smatrix.set(2, 2, 6);
    ss << smatrix;
    ASSERT_EQ(ss.str(), "1 2 3 \n2 4 5 \n3 5 6 \n") << "wrong result" ;
}
/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:
1. Declare an empty matrix with the default constructor for MatrixSymmetric.
Assert that the NormL0 and size functions return appropriate values for these.
2. Using the second constructor that takes size as argument, create a matrix of
size zero. Repeat the assertions from (1).
3. Provide a negative argument to the second constructor and assert that the
constructor throws an exception.
4. Create and initialize a matrix of some size, and verify that the NormL0
function returns the correct value.
5. Create a matrix, initialize some or all of its elements, then retrieve and
check that they are what you initialized them to.
6. Create a matrix of some size. Make an out-of-bounds access into it and check
that an exception is thrown.
7. Test the stream operator using std::stringstream and using the "<<" operator.

*/
