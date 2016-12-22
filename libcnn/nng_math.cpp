#include "nng_math.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

const double epsilon = 0.000001;

void nng::print_v(const Vectord& v)
{
  for (size_t i=0; i<v.size(); i++)
    std::cout<<v[i]<<" ";
  std::cout<<std::endl;
}

double nng::rand_a_b(double a, double b)
{
    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}

double nng::normal_distribution_rand(double mean, double stddev)
{
    std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, stddev);
	return distribution(generator);
}

double nng::sigmoid(const double x)
{
  return 1 / (1 + exp(-x));
}

double nng::sigmoid_prime(const double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

double nng::kl_divergence(const double x, const double y)
{
  return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y));
}

nng::Matrix2d operator+(double scalar, nng::Matrix2d& matrix) 
{
    return matrix + scalar;
}
nng::Matrix2d operator-(double scalar, nng::Matrix2d& matrix) 
{
    size_t rows = matrix.get_rows();
    size_t cols = matrix.get_cols();
    nng::Matrix2d m_scalar(rows,cols,scalar);
    return m_scalar - matrix;
}

nng::Matrix2d operator*(double scalar, nng::Matrix2d& matrix) 
{
    return matrix * scalar;
}

nng::Vector operator+(double scalar, nng::Vector& v) 
{
    return v + scalar;
}
nng::Vector operator-(double scalar, nng::Vector& v) 
{
    size_t len = v.get_length();
    nng::Vector v_scalar(len,scalar);
    return v_scalar - v;
}

nng::Vector operator*(double scalar, nng::Vector& v) 
{
    return v * scalar;
}

nng::Vector nng::column_vector_to_cnn_vector(const nng::column_vector& x)
{
    size_t len = x.nr();
    nng::Vector v(len,0);
    for(size_t i = 0; i < len; ++i)
    {
        v(i) = x(i);
    }	
	return v;
}

nng::column_vector nng::cnn_vector_to_column_vector(const nng::Vector& v)
{
	size_t len = v.get_length();
    nng::column_vector x(len);
    for(size_t i = 0; i < len; ++i)
    {
        x(i) = v(i);
    }
    return x;
}