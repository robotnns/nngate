#include "Matrix2d.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>
#include "nng_math.h"

using namespace nng;
using namespace std;

Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4) 
{
  _mat.resize(s1);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
    _mat[i].resize(s2);
	for (size_t j = 0; j < s2; ++j)
	{
	  _mat[i][j] = Matrix2d(s3, s4);
	}
		
  }
  
  _shape.resize(4);
  _shape[0] = s1;
  _shape[1] = s2;
  _shape[2] = s3;
  _shape[3] = s4;
}

Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const double& initial) 
{
  _mat.resize(s1);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
    _mat[i].resize(s2);
	for (size_t j = 0; j < s2; ++j)
	{
	  _mat[i][j] = Matrix2d(s3, s4, initial);
	}
		
  }
  
  _shape.resize(4);
  _shape[0] = s1;
  _shape[1] = s2;
  _shape[2] = s3;
  _shape[3] = s4;
}

// Conversiont from vector to matrix, row major
Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const Vectord& vec)
{
  assert (vec.size() == s1*s2*s3*s4);
  nng::Vector v(vec);
  Matrix4d(s1, s2, s3, s4, v);
}

Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const nng::Vector& v)
{
  assert (v.get_length() == s1*s2*s3*s4);
  _mat.resize(s1);
  Matrix2d m(s3, s4);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
    _mat[i].resize(s2);
	for (size_t j = 0; j < s2; ++j)
	{
	  m = Matrix2d(s3,s4,v.getSegment(i*s2*s3*s4 + j*s3*s4,s3*s4));
	  _mat[i][j] = m;
	}
  }

  _shape.resize(4);
  _shape[0] = s1;
  _shape[1] = s2;
  _shape[2] = s3;
  _shape[3] = s4;
}

Matrix2d::Matrix4d(const Matrix4d& rhs) 
{
  _mat = rhs._mat;
  _shape = rhs.shape();
}

Matrix4d::Matrix4d(Matrix4d&& rhs)   
{
  _mat = rhs._mat;
  _shape = rhs.shape();
  rhs._mat.clear();
}

Matrix4d& Matrix4d::operator=(Matrix4d&& rhs)
{
	if (this!=&rhs)
	{
		// pilfer other’s resource
		_mat = rhs._mat;
		_shape = rhs.shape();
		// reset other
		rhs._mat.clear();
	}
	return *this;
}

Matrix4d::~Matrix4d() {}

// Assignment Operator
Matrix4d& Matrix4d::operator=(const Matrix4d& rhs) 
{
	if (this!=&rhs)
	{
		// pilfer other’s resource
		_mat = rhs._mat;
		_shape = rhs.shape();
	}
	return *this;
}

Vectord Matrix2d::get_col(size_t index)
{
	Vectord result;
    for (size_t i=0; i<_rows; ++i)
	{
		result.push_back(_mat[i][index]);
	}
	return result;
}

void Matrix2d::set_col(const Vectord& v, size_t index)
{
	assert (v.size() == _rows);
    for (size_t i=0; i<_rows; ++i)
	{
		_mat[i][index] = v[i];
	}
}

void Matrix2d::set_col(const nng::Vector& v, size_t index)
{
	assert (v.get_length() == _rows);
    for (size_t i=0; i<_rows; ++i)
	{
		_mat[i][index] = v[i];
	}
}

// Matrix/scalar addition
Matrix2d Matrix2d::operator+(const double& rhs) 
{
  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] + rhs;
    }
  }

  return result;
}

// Matrix/scalar subtraction
Matrix2d Matrix2d::operator-(const double& rhs) 
{
  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] - rhs;
    }
  }

  return result;
}


// Matrix/scalar multiplication
Matrix2d Matrix2d::operator*(const double& rhs) 
{
  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] * rhs;
    }
  }

  return result;
}

// Matrix/scalar division
Matrix2d Matrix2d::operator/(const double& rhs) 
{
  assert (std::abs(rhs) > epsilon);

  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] / rhs;
    }
  }

  return result;
}

// Multiply a matrix with a vector
Vectord Matrix2d::operator*(const Vectord& rhs) 
{
  assert (_cols == rhs.size());

  Vectord result(_rows, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result[i] += _mat[i][j] * rhs[j];
    }
  }

  return result;
}

// divide each m[i][j] by v[j]
Matrix2d Matrix2d::operator/(const nng::Vector& rhs)
{
	assert (_cols == rhs.get_length());
	Matrix2d result(_rows, _cols, 0.0);
	for (size_t i=0; i<_rows; ++i) 
	{
	for (size_t j=0; j<_cols; ++j) 
	{
	  result(i,j) = _mat[i][j] / rhs[j];
	}
	}

	return result;
}

// Convert from matrix to vector
Vectord Matrix2d::toVector()
{
    Vectord v;
    for (size_t i = 0; i < _rows; ++i)
    {
        v.insert(v.end(),_mat[i].begin(), _mat[i].end());
    }
    return v;
}

nng::Vector Matrix2d::to_cnnvector()
{
    return nng::Vector(this->toVector());
}

// Access the individual elements
double& Matrix2d::operator()(const size_t& row, const size_t& col) 
{
  return _mat[row][col];
}

// Access the individual elements (const)
const double& Matrix2d::operator()(const size_t& row, const size_t& col) const 
{
  return _mat[row][col];
}

void Matrix2d::setElement(double value, const size_t& row, const size_t& col)
{
    _mat[row][col] = value;
}

void Matrix2d::print()
{
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      std::cout << _mat[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

Identity::Identity(size_t n):Matrix2d(n,n,0)
{
    for (size_t i = 0; i < n; ++i)
    {
        setElement(1.0, i,i);
    }
}