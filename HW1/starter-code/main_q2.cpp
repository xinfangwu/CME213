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

// 0. test the functions in Matrix2D
TEST(testMatrix, case0Test){
  // empty matrix 
  Matrix2D<int> A;
  ASSERT_EQ(A.size_rows(), 0) << "empty matrix's size_rows wrong";
  ASSERT_EQ(A.size_cols(), 0) << "empty matrix's size_cols wrong";

  Matrix2D<int> B(3, 2);
  ASSERT_EQ(B.size_rows(), 3) << "matrix's size_rows wrong";
  ASSERT_EQ(B.size_cols(), 2) << "matrix's size_cols wrong";

  B(0,0) = 1;
  B(1,0) = 2;
  B(2,0) = 3;
  B(0,1) = 2;
  B(1,1) = 4;
  B(2,1) = 6;
  EXPECT_THROW(B(-1, 100), std::out_of_range) << "Access out of range";
  EXPECT_THROW(B(3, 3), std::out_of_range) << "Access out of range";

  std::stringstream ss;
  ss << A;
  ASSERT_EQ(ss.str(), "") << "print wrong A" ;
  std::stringstream ss2;
  ss2 << B;
  ASSERT_EQ(ss2.str(), "1 2 \n2 4 \n3 6 \n") << "print wrong B" ;
}

// 1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
TEST(testMatrix, case1Test){
  Matrix2D<int> A(2, 2);
  Matrix2D<int> B(3, 3);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(1,0) = 5;
  A(1,1) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(1,0) = 1;
  B(1,1) = 1;
  B(1,2) = 1;
  B(2,0) = 1;
  B(2,1) = 1;
  B(2,2) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

TEST(testMatrix, case2Test){
  Matrix2D<int> A(2, 3);
  Matrix2D<int> B(2, 3);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;
  A(1,0) = 5;
  A(1,1) = 5;
  A(1,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(1,0) = 1;
  B(1,1) = 1;
  B(1,2) = 1;

  Matrix2D<int> exp(2, 3);
  exp(0,0) = 5;
  exp(0,1) = 5;
  exp(0,2) = 5;
  exp(1,0) = 5;
  exp(1,1) = 5;
  exp(1,2) = 5;

  Matrix2D<int> result = A.dot(B);
  std::stringstream ss_dot;
  ss_dot << result;
  std::stringstream ss_exp;
  ss_exp << exp;
  ASSERT_EQ(ss_dot.str(), ss_exp.str()) << "A.dot(B) is wrong"; 
}

TEST(testMatrix, case3Test){
  Matrix2D<int> A(2, 3);
  Matrix2D<int> B(2, 4);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;
  A(1,0) = 5;
  A(1,1) = 5;
  A(1,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(0,3) = 1;
  B(1,0) = 1;
  B(1,1) = 1;
  B(1,2) = 1;
  B(1,3) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

TEST(testMatrix, case4Test){
  Matrix2D<int> A(2, 3);
  Matrix2D<int> B(3, 4);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;
  A(1,0) = 5;
  A(1,1) = 5;
  A(1,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(0,3) = 1;
  B(1,0) = 1;
  B(1,1) = 1;
  B(1,2) = 1;
  B(1,3) = 1;
  B(2,0) = 1;
  B(2,1) = 1;
  B(2,2) = 1;
  B(2,3) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

// 2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
// TODO
TEST(testMatrix, case5Test){
  Matrix2D<int> A(1, 3);
  Matrix2D<int> B(3, 4);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(0,3) = 1;
  B(1,0) = 1;
  B(1,1) = 1;
  B(1,2) = 1;
  B(1,3) = 1;
  B(2,0) = 1;
  B(2,1) = 1;
  B(2,2) = 1;
  B(2,3) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

// 3. A of shape (1, n != 1), B of shape (1, n != 1)
TEST(testMatrix, case6Test){
  Matrix2D<int> A(1, 3);
  Matrix2D<int> B(1, 4);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;
  B(0,3) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

TEST(testMatrix, case7Test){
  Matrix2D<int> A(1, 3);
  Matrix2D<int> B(1, 3);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;

  B(0,0) = 1;
  B(0,1) = 1;
  B(0,2) = 1;

  Matrix2D<int> exp(1, 3);
  exp(0,0) = 5;
  exp(0,1) = 5;
  exp(0,2) = 5;

  Matrix2D<int> result = A.dot(B);
  std::stringstream ss_dot;
  ss_dot << result;
  std::stringstream ss_exp;
  ss_exp << exp;
  ASSERT_EQ(ss_dot.str(), ss_exp.str()) << "A.dot(B) is wrong"; 
}


// 3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
TEST(testMatrix, case8Test){
  Matrix2D<int> A(3, 1);
  Matrix2D<int> B(4, 1);  
  A(0,0) = 5;
  A(1,0) = 5;
  A(2,0) = 5;

  B(0,0) = 1;
  B(1,0) = 1;
  B(2,0) = 1;
  B(3,0) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

TEST(testMatrix, case9Test){
  Matrix2D<int> A(2, 3);
  Matrix2D<int> B(4, 1);  
  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;
  A(1,0) = 5;
  A(1,1) = 5;
  A(1,2) = 5;
  

  B(0,0) = 1;
  B(1,0) = 1;
  B(2,0) = 1;
  B(3,0) = 1;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}
// 4. A of shape (1, 1), B of shape (m != 1, n != 1)
TEST(testMatrix, case10Test){
  Matrix2D<int> A(1, 1);
  Matrix2D<int> B(2, 2);

  A(0,0) = 5;
  B(0,0) = 1;
  B(1,0) = 1;
  B(0,1) = 1;
  B(1,1) = 1;

  Matrix2D<int> exp(2, 2);
  exp(0, 0) = 5;
  exp(0, 1) = 5;
  exp(1, 0) = 5;
  exp(1, 1) = 5;

  Matrix2D<int> result = A.dot(B);
  std::stringstream ss_dot;
  ss_dot << result;
  std::stringstream ss_exp;
  ss_exp << exp;
  ASSERT_EQ(ss_dot.str(), ss_exp.str()) << "A.dot(B) is wrong";
}

TEST(testMatrix, case11Test){
  Matrix2D<int> A(1, 1);
  Matrix2D<int> B(3, 1);

  A(0,0) = 5;
  B(0,0) = 1;
  B(1,0) = 1;
  B(2,0) = 1;


  Matrix2D<int> exp(3, 1);
  exp(0, 0) = 5;
  exp(1, 0) = 5;
  exp(2, 0) = 5;
  
  Matrix2D<int> result = A.dot(B);
  std::stringstream ss_dot;
  ss_dot << result;
  std::stringstream ss_exp;
  ss_exp << exp;
  ASSERT_EQ(ss_dot.str(), ss_exp.str()) << "A.dot(B) is wrong";
}

TEST(testMatrix, case12Test){
  Matrix2D<int> A(1, 3);
  Matrix2D<int> B(3, 1);

  A(0,0) = 5;
  A(0,1) = 5;
  A(0,2) = 5;

  B(0,0) = 1;
  B(1,0) = 1;
  B(2,0) = 1;

  Matrix2D<int> exp(3, 3);
  exp(0, 0) = 5;
  exp(0, 1) = 5;
  exp(0, 2) = 5;
  exp(1, 0) = 5;
  exp(1, 1) = 5;
  exp(1, 2) = 5;
  exp(2, 0) = 5;
  exp(2, 1) = 5;
  exp(2, 2) = 5;
  
  Matrix2D<int> result = A.dot(B);
  std::stringstream ss_dot;
  ss_dot << result;
  std::stringstream ss_exp;
  ss_exp << exp;
  ASSERT_EQ(ss_dot.str(), ss_exp.str()) << "A.dot(B) is wrong";
}

// 5. empty case
TEST(testMatrix, case13Test){
  Matrix2D<int> A(1, 1);
  Matrix2D<int> B(0, 0);

  A(0,0) = 5;

  EXPECT_THROW(A.dot(B), std::invalid_argument);
}

/*
TODO:
Test your implementation by writing tests that cover most scenarios of 2D matrix
broadcasting. Say you are testing the result C = A * B, test with:
1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
4. A of shape (1, 1), B of shape (m != 1, n != 1)
Please test any more cases that you can think of.
*/
