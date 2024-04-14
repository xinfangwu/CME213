#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <cmath>

// TODO: add your function here. The function should count the number of
// entries between lb and ub.
unsigned int count_range(const std::set<double>& data, const double lb,
                         const double ub) 
{
  if (lb >= ub){
    throw std::invalid_argument("lower bound is equal or larger than upper bound");
  }
  else{
    auto lower = data.lower_bound(lb);
    auto upper = data.upper_bound(ub);
    return std::distance(lower, upper);
  }

}

int main() {
  std::set<double> data_simple{0, 1, 2, 3, 4, 5, 6};

  // Range test
  try {
    count_range(data_simple, 1, 0);
    std::cout << "Error: range test." << std::endl;
  } catch (const std::exception& error) {
    // This line is expected to be run
    std::cout << "Range test passed." << std::endl;
  }

  // Count test
  assert(count_range(data_simple, 3, 6) == 4);
  
  // Test with N(0,1) data.
  std::set<double> data_rng;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  unsigned int n = 10000;
  float p = 0.6827;
  for (unsigned int i = 0; i < n; ++i) data_rng.insert(distribution(generator));

  std::cout << "X = number of elements in range [-1, 1]: "
            << count_range(data_rng, -1, 1)
            << " (approx. 99% confidence interval for X: (" 
            << n * p - 2.576 * sqrt(n * p * (1 - p)) << ", "
            << n * p + 2.576 * sqrt(n * p * (1 - p)) << "))"
            << std::endl;

  return 0;
}
