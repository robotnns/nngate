#include "Vector.h"
#include "nng_math.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

using namespace nng;
using namespace std;
Vector::Vector(size_t len, const double& initial)
{
    vec.resize(len);
    for (size_t i=0; i<len; ++i)
        vec[i] = initial;
    length = len;
}


Vector::Vector(const Vectord& v):
vec(v)
{
    length = v.size();
}

Vector::Vector(const Vector& rhs) {
  vec = rhs.vec;
  length = rhs.get_length();
}

// Access the individual elements
double& Vector::operator()(const size_t& index)
{
    return vec[index];
}

const double& Vector::operator()(const size_t& index) const
{
    return vec[index];
}

double& Vector::operator[](const size_t& index)
{
    return vec[index];
}

const double& Vector::operator[](const size_t& index) const
{
    return vec[index];
}

//Vector/Vector addition
Vector Vector::operator+(const Vector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] + rhs(i);
  }

  return result;

}

//Vector/Vector subtraction
Vector Vector::operator-(const Vector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] - rhs(i);
  }

  return result;

}

//Vector/Vector division
Vector Vector::operator/(const Vector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] / rhs(i);
  }

  return result;

}

// Vector/Vector dot product
Vector Vector::dot(const Vector& v)
{
  // check dimension
  size_t len = v.get_length();
  assert (length == len);

  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) 
  {
      result(i) = vec[i]* v(i);
  }

  return result;

}

// Vector/scalar addition
Vector Vector::operator+(const double& rhs) 
{
  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] + rhs;
  }

  return result;
}

// Vector/scalar subtraction
Vector Vector::operator-(const double& rhs) 
{
  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] - rhs;
  }

  return result;
}

// unary - operator
Vector Vector::operator-() const
{
  Vector result(length, 0.0);
  for (size_t i=0; i<length; ++i) {
      result(i) = -vec[i];
  }

  return result;
}

// Vector/scalar multiplication
Vector Vector::operator*(const double& rhs) 
{
  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] * rhs;
  }

  return result;
}
// Vector/scalar division
Vector Vector::operator/(const double& rhs) 
{
  assert (std::abs(rhs) > epsilon);

  Vector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] / rhs;
  }

  return result;
}

void Vector::print()
{
  for (size_t i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl;
}

// Get submatrix of length = len starting at start_index
Vector Vector::getSegment(size_t start_index, size_t len) const
{
  assert (start_index < length && start_index + len < length);
  Vector result(len, 0.0);

  for (size_t i=0; i<len; ++i) {
      result(i) = vec[start_index+i];
  }

  return result;
}

void Vector::setSegment(size_t start_index, size_t len, const Vector& v)
{
  for (size_t i=0; i<len; ++i) {
      vec[start_index+i] = v(i);
  }
}


Vector Vector::concatenate(const Vectord& v)
{
    Vector result(*this);
    result.getVector().insert(result.getVector().end(),v.begin(), v.end());
    result.setLength(length + v.size());
    return result;
}

Vector Vector::concatenate(const Vector& v)
{
    Vector result(*this);
    result.getVector().insert(result.getVector().end(),v.getVector().begin(), v.getVector().end());
    result.setLength(length + v.get_length());
    return result;
}

// kl divergence between each element in the current vector and v
Vector Vector::kl_divergence(Vector& v)
{
    assert (length == v.get_length());
    Vector result(*this);
    double sum_vec = sum();
    double sum_v = v.sum();

    for (size_t i=0; i<length; ++i) {
        result(i) = nng::kl_divergence(vec[i]/sum_vec, v(i)/sum_v);
    }
    return result;
}

// sum of all the elements in the vector
double Vector::sum()
{
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}