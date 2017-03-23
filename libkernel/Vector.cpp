#include "Vector.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

using namespace nng;
using namespace std;

Vector::Vector(size_t len)
{
    _vec.resize(len);
    _length = len;
}

Vector::Vector(size_t len, const double& initial)
{
    _vec.resize(len, initial);
    _length = len;
}


Vector::Vector(const Vectord& v):
_vec(v)
{
    _length = v.size();
}

Vector::Vector(const Vector& rhs) 
{
  _vec = rhs._vec;
  _length = rhs.get_length();
}

Vector::Vector(Vector&& rhs)   
{
  _vec = rhs._vec;
  _length = rhs.get_length();
  rhs._vec.clear();
}

Vector& Vector::operator=(Vector&& rhs)
{
	if (this!=&rhs)
	{
		// pilfer other’s resource
		_vec = rhs._vec;
		_length = rhs.get_length();
		// reset other
		rhs._vec.clear();
	}
	return *this;
}

// Access the individual elements
double& Vector::operator()(const size_t& index)
{
    return _vec[index];
}

const double& Vector::operator()(const size_t& index) const
{
    return _vec[index];
}

double& Vector::operator[](const size_t& index)
{
    return _vec[index];
}

const double& Vector::operator[](const size_t& index) const
{
    return _vec[index];
}

//Vector/Vector addition
Vector Vector::operator+(const Vector& rhs)
{
  // check dimension
  assert (_length == rhs.get_length());

  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] + rhs(i);
  }

  return result;

}

//Vector/Vector subtraction
Vector Vector::operator-(const Vector& rhs)
{
  // check dimension
  assert (_length == rhs.get_length());

  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] - rhs(i);
  }

  return result;

}

//Vector/Vector division
Vector Vector::operator/(const Vector& rhs)
{
  // check dimension
  assert (_length == rhs.get_length());

  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] / rhs(i);
  }

  return result;

}

// Vector/Vector dot product
Vector Vector::dot(const Vector& v)
{
  // check dimension
  size_t len = v.get_length();
  assert (_length == len);

  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) 
  {
      result(i) = _vec[i]* v(i);
  }

  return result;

}

// Vector/scalar addition
Vector Vector::operator+(const double& rhs) 
{
  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] + rhs;
  }

  return result;
}

// Vector/scalar subtraction
Vector Vector::operator-(const double& rhs) 
{
  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] - rhs;
  }

  return result;
}

// unary - operator
Vector Vector::operator-() const
{
  Vector result(_length, 0.0);
  for (size_t i=0; i<_length; ++i) {
      result(i) = -_vec[i];
  }

  return result;
}

// Vector/scalar multiplication
Vector Vector::operator*(const double& rhs) 
{
  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] * rhs;
  }

  return result;
}
// Vector/scalar division
Vector Vector::operator/(const double& rhs) 
{
  assert (std::abs(rhs) > epsilon);

  Vector result(_length, 0.0);

  for (size_t i=0; i<_length; ++i) {
      result(i) = _vec[i] / rhs;
  }

  return result;
}

void Vector::print()
{
  for (size_t i=0; i<_vec.size(); i++)
    std::cout<<_vec[i]<<" ";
  std::cout<<std::endl;
}

// Get submatrix of _length = len starting at start_index
Vector Vector::getSegment(size_t start_index, size_t len) const
{
  assert (start_index < _length && start_index + len <= _length);
  Vector result(len, 0.0);

  for (size_t i=0; i<len; ++i) {
      result(i) = _vec[start_index+i];
  }

  return result;
}

// Get submatrix starting at start_index to the end
Vector Vector::getTail(size_t start_index) const
{
  assert (start_index < _length);
  Vector result(_length-start_index, 0.0);

  for (size_t i=start_index; i<_length; ++i) {
      result(i-start_index) = _vec[i];
  }

  return result;
}

void Vector::setSegment(size_t start_index, size_t len, const Vector& v)
{
  for (size_t i=0; i<len; ++i) {
      _vec[start_index+i] = v(i);
  }
}


Vector Vector::concatenate(const Vectord& v)
{
    Vector result(*this);
    result.getVector().insert(result.getVector().end(),v.begin(), v.end());
    result.setLength(_length + v.size());
    return result;
}

Vector Vector::concatenate(const Vector& v)
{
    Vector result(*this);
    result.getVector().insert(result.getVector().end(),v.getVector().begin(), v.getVector().end());
    result.setLength(_length + v.get_length());
    return result;
}


// sum of all the elements in the vector
double Vector::sum()
{
    return std::accumulate(_vec.begin(), _vec.end(), 0.0);
}

// average of all the elements in the vector
double Vector::mean()
{
    return this->sum()/_length;
}

// power at each element
Vector Vector::power(const double exponent)
{
  Vector result(*this);
  for (size_t i=0; i<_length; ++i) 
  {
      result(i) = pow(_vec[i], exponent);
  }
  return result;
}

// sqrt at each element
Vector Vector::sqrt()
{
  Vector result(*this);
  for (size_t i=0; i<_length; ++i) 
  {
      result(i) = std::sqrt(_vec[i]);
  }
  return result;
}

//
double Vector::norm2()
{
  Vector v_square = this->power(2);
  double result = std::sqrt(v_square.sum());
  return result;	
}

// normalize the vector
Vector Vector::normalize()
{
  Vector result(_length, 0.0);
  double sum = this->norm2();
  if (sum<epsilon) sum = epsilon;
  for (size_t i=0; i<_length; ++i) 
  {
      result(i) = _vec[i]/sum;
  }
  return result;	
}

Matrix2d Vector::toDiagonal()
{
	Matrix2d result(_length, _length);
	for (size_t i=0; i<_length; ++i)
		result(i,i) = _vec[i];
	return result;
}
