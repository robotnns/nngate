#include "Matrix2d.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>
#include "Vector.h"
#include "nng_math.h"

using namespace nng;
using namespace std;

Matrix2d::Matrix2d(size_t r, size_t c) 
{
  _mat.resize(r);
  for (size_t i=0; i<_mat.size(); ++i) {
    _mat[i].resize(c);
  }
  _rows = r;
  _cols = c;
}

Matrix2d::Matrix2d(size_t r, size_t c, const double& initial) 
{
  _mat.resize(r);
  for (size_t i=0; i<_mat.size(); ++i) {
    _mat[i].resize(c, initial);
  }
  _rows = r;
  _cols = c;
}

// Conversiont from vector to matrix, row major
Matrix2d::Matrix2d(size_t r, size_t c, const Vectord& v)
{
  assert (v.size() == r*c);
  _mat.resize(r);
  for (size_t i = 0; i <= v.size() - c ; i += c){
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + c;
    _mat[i/c].assign(first, last);
  }
  _rows = r;
  _cols = c;
}

Matrix2d::Matrix2d(size_t r, size_t c, const nng::Vector& cnnv)
{
  Vectord v = cnnv.getVector();
  assert (v.size() == r*c);
  _mat.resize(r);
  for (size_t i = 0; i <= v.size() - c ; i += c){
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + c;
    _mat[i/c].assign(first, last);
  }
  _rows = r;
  _cols = c;
}

Matrix2d::Matrix2d(const Matrix2d& rhs) 
{
  _mat = rhs._mat;
  _rows = rhs.get_rows();
  _cols = rhs.get_cols();
}

Matrix2d::Matrix2d(Matrix2d&& rhs)   
{
  _mat = rhs._mat;
  _rows = rhs.get_rows();
  _cols = rhs.get_cols();
  rhs._mat.clear();
}

Matrix2d& Matrix2d::operator=(Matrix2d&& rhs)
{
	if (this!=&rhs)
	{
		// pilfer other’s resource
		_mat = rhs._mat;
		_rows = rhs.get_rows();
		_cols = rhs.get_cols();
		// reset other
		rhs._mat.clear();
	}
	return *this;
}

Matrix2d::~Matrix2d() {}

// Assignment Operator
Matrix2d& Matrix2d::operator=(const Matrix2d& rhs) 
{
  if (&rhs == this)
    return *this;

  size_t new_rows = rhs.get_rows();
  size_t new_cols = rhs.get_cols();

  _mat.resize(new_rows);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
    _mat[i].resize(new_cols);
  }

  for (size_t i=0; i<new_rows; ++i) 
  {
    for (size_t j=0; j<new_cols; ++j) 
	{
      _mat[i][j] = rhs(i, j);
    }
  }
  _rows = new_rows;
  _cols = new_cols;

  return *this;
}

// Addition of two matrices
Matrix2d Matrix2d::operator+(const Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_rows == rhs_rows && _cols == rhs_cols);

  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] + rhs(i,j);
    }
  }

  return result;
}

// Cumulative addition of this matrix and another
Matrix2d& Matrix2d::operator+=(const Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_rows == rhs_rows && _cols == rhs_cols);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      _mat[i][j] += rhs(i,j);
    }
  }

  return *this;
}


// Subtraction of this matrix and another
Matrix2d Matrix2d::operator-(Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_rows == rhs_rows && _cols == rhs_cols);

  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] - rhs(i,j);
    }
  }

  return result;
}

Matrix2d Matrix2d::operator-(const Matrix2d& rhs) const
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_rows == rhs_rows && _cols == rhs_cols);

  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] - rhs(i,j);
    }
  }

  return result;
}

// Cumulative subtraction of this matrix and another
Matrix2d& Matrix2d::operator-=(const Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_rows == rhs_rows && _cols == rhs_cols);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      _mat[i][j] -= rhs(i,j);
    }
  }

  return *this;
}


// Left multiplication of this matrix and another
Matrix2d Matrix2d::operator*(const Matrix2d& rhs) 
{
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (_cols == rhs_rows);

  Matrix2d result(_rows, rhs_cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<rhs_cols; ++j) 
	{
      for (size_t k=0; k<_cols; k++) 
	  {
        result(i,j) += _mat[i][k] * rhs(k,j);
      }
    }
  }

  return result;
}

// Cumulative left multiplication of this matrix and another
Matrix2d& Matrix2d::operator*=(const Matrix2d& rhs) 
{
  Matrix2d result = (*this) * rhs;
  (*this) = result;
  return *this;
}

// unary - operator
Matrix2d Matrix2d::operator-() const
{
  Matrix2d result(_rows, _cols, 0.0);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = -_mat[i][j];
    }
  }

  return result;
}

// Matrix/Matrix dot product
Matrix2d Matrix2d::dot(const Matrix2d& m)
{
  // check dimension
  size_t m_rows = m.get_rows();
  size_t m_cols = m.get_cols();
  assert (_rows == m_rows && _cols == m_cols);

  Matrix2d result(_rows, _cols, 0.0);

  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = _mat[i][j] * m(i,j);
    }
  }

  return result;

}

// Calculate a transpose of this matrix

Matrix2d Matrix2d::transpose() 
{
  Matrix2d result(_cols, _rows, 0.0);

  for (size_t i=0; i<_cols; ++i) 
  {
    for (size_t j=0; j<_rows; ++j) 
	{
      result(i,j) = _mat[j][i];
    }
  }

  return result;
}

const Matrix2d Matrix2d::transpose() const
{
  Matrix2d result(_cols, _rows, 0.0);

  for (size_t i=0; i<_cols; ++i) 
  {
    for (size_t j=0; j<_rows; ++j) 
	{
      result(i,j) = _mat[j][i];
    }
  }

  return result;
}

/*
Matrix2d& Matrix2d::transpose()
{
  double temp = 0;
  Vectord v = this->toVector();
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=i+1; j<_cols; ++j) 
	{
      	temp = v[i*_cols+j];
	v[i*_cols+j] = v[j*_cols+i];
        v[j*_cols+i] = temp;
    }
  }
  int r_temp = _rows;
  _rows = _cols;
  _cols = r_temp;
  _mat.resize(_rows);
  for (size_t i = 0; i <= v.size() - _cols ; i += _cols)
  {
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + _cols;
    _mat[i/_cols].assign(first, last);
 } 
 return *this;
}
*/
// Get submatrix of size (r,c) starting at indices (i,j)
Matrix2d Matrix2d::getBlock(size_t i, size_t j, size_t r, size_t c) const 
{
  Matrix2d result(r, c, 0.0);

  for (size_t p=0; p<r; ++p) 
  {
      for (size_t q=0; q<c; ++q) 
	  {
          result(p,q) = _mat[i+p][j+q];
      }
  }

  return result;
}


void Matrix2d::setBlock(size_t i, size_t j, size_t r, size_t c, const Matrix2d& m)
{

  for (size_t p=0; p<r; ++p) 
  {
      for (size_t q=0; q<c; ++q) 
	  {
          _mat[i+p][j+q] = m(p,q);
      }
  }
}

// Concatenation with matrix m
// param: axis = 0 concatenate horizontally
//        axis = 1 concatenate vertically
Matrix2d Matrix2d::concatenate(const Matrix2d& m, size_t axis)
{
  assert (axis == 0 || axis == 1);
  size_t m_rows = m.get_rows();
  size_t m_cols = m.get_cols();

  if (axis == 0)
  {
      assert (_rows == m_rows);

      Matrix2d result(_rows, _cols + m_cols, 0.0);
      result.setBlock(0,0,_rows,_cols, *this);
      result.setBlock(0,_cols,_rows,m_cols, m);

      return result;
  }
  else
  {
      assert (_cols == m_cols);

      Matrix2d result(_rows + m_rows, _cols, 0.0);
      result.setBlock(0,0,_rows,_cols, *this);
      result.setBlock(_rows,0,m_rows,_cols, m);

      return result;
  }
}

// sigmoid at each element
Matrix2d Matrix2d::sigmoid()
{
  Matrix2d result(*this);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = nng::sigmoid(_mat[i][j]);
    }
  }
  return result;
}

// sum over _rows or columns
nng::Vector Matrix2d::sum(size_t axis)
{
  assert (axis == 0 || axis == 1);
  if (axis == 0) // sum over _rows for each column
  {
      nng::Vector result(_cols,0);
      for (size_t j=0; j<_cols; ++j)
          for (size_t i=0; i<_rows; ++i)
              result(j) += _mat[i][j];
      return result;
  }
  else // sum over _cols for each row
  {
      nng::Vector result(_rows,0);
      for (size_t i=0; i<_rows; ++i) 
	  {
        for (size_t j=0; j<_cols; ++j) 
		{
          result(i) += _mat[i][j];
        }
      }
      return result;
  }
}

// sum over entire matrix
double Matrix2d::sum()
{
  double result = 0;
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result += _mat[i][j];
    }
  }
  return result;
}

// power at each element
Matrix2d Matrix2d::power(const double exponent)
{
  Matrix2d result(*this);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = pow(_mat[i][j], exponent);
    }
  }
  return result;
}

// derivation of sigmoid at each element
Matrix2d Matrix2d::sigmoid_prime()
{
  Matrix2d result(*this);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = nng::sigmoid_prime(_mat[i][j]);
    }
  }
  return result;
}

// max element
double Matrix2d::max()
{
	Vectord v = toVector();
	double max = *std::max_element(v.begin(), v.end());
	return max;
}

// min element
double Matrix2d::min()
{
	Vectord v = toVector();
	double min = *std::min_element(v.begin(), v.end());
	return min;
}

Matrix2d Matrix2d::exp()
{
  Matrix2d result(*this);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = std::exp((long double)_mat[i][j]);
    }
  }
  return result;
}

Matrix2d Matrix2d::log()
{
  Matrix2d result(_rows, _cols, 0.0);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = std::log((long double)_mat[i][j]);
    }
  }
  return result;
}

Matrix2d Matrix2d::abs()
{
  Matrix2d result(_rows, _cols, 0.0);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = std::abs((long double)_mat[i][j]);
    }
  }
  return result;
} 

Matrix2d Matrix2d::floor()
{
  Matrix2d result(_rows, _cols, 0.0);
  for (size_t i=0; i<_rows; ++i) 
  {
    for (size_t j=0; j<_cols; ++j) 
	{
      result(i,j) = std::floor((long double)_mat[i][j]);
    }
  }
  return result;
} 

nng::Vector Matrix2d::argmax(size_t axis)
{
	assert (axis == 0 || axis == 1);
	if (axis == 0) // each element in result is argmax of the column vector of the matrix
	{
		nng::Vector result(_cols,0);
		Vectord v;
		size_t arg_max;
		for (size_t i=0; i<_cols; ++i)
		{
			v = get_col(i);
			arg_max = std::distance(v.begin(), std::max_element(v.begin(), v.end()));
			result(i) = arg_max;
		}
		return result;
	}
	else
	{
		nng::Vector result(_rows,0);
		Vectord v;
		size_t arg_max;
		for (size_t i=0; i<_rows; ++i)
		{
			v = _mat[i];
			arg_max = std::distance(v.begin(), std::max_element(v.begin(), v.end()));
			result(i) = arg_max;
		}
		return result;
	}
}

bool Matrix2d::isUpperTriangle(double eps)
{
    if (_cols != _rows || _cols < 1) 
        return false;
    if (_cols == 1) 
        return true;

    double sum = 0.0;
    for(size_t i = 0; i < _cols; ++i)
	{
        for(size_t j=i+1; j<_rows; ++j)
            sum += std::abs(_mat[j][i]);
			
	}

    if (sum < eps)
        return true;
    else
        return false;
}

bool Matrix2d::isLowerTriangle(double eps)
{
    if (_cols != _rows || _cols < 1) 
        return false;
    if (_rows == 1) 
        return true;

    double sum = 0.0;
    for(size_t i = 1; i < _rows; ++i)
        for(size_t j=i+1; j<_cols; ++j)
            sum += std::abs(_mat[i][j]);
    if (sum < eps)
        return true;
    else
        return false;
}

bool Matrix2d::isDiagonal(double eps)
{
    return isUpperTriangle(eps) && isLowerTriangle(eps);
}
        
nng::Vector Matrix2d::getDiagonal()
{
    assert(_cols == _rows);
    Vectord result;
    for(size_t i = 0; i < _rows; ++i)
	{
        result.push_back(_mat[i][i]);
	}
    return nng::Vector(result);
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