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
Matrix2d::Matrix2d(size_t r, size_t c, const double& initial) 
{
  mat.resize(r);
  for (size_t i=0; i<mat.size(); ++i) {
    mat[i].resize(c, initial);
  }
  rows = r;
  cols = c;
}

// Conversiont from vector to matrix, row major
Matrix2d::Matrix2d(size_t r, size_t c, const Vectord& v)
{
  assert (v.size() == r*c);
  mat.resize(r);
  for (size_t i = 0; i <= v.size() - c ; i += c){
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + c;
    mat[i/c].assign(first, last);
  }
  rows = r;
  cols = c;
}

Matrix2d::Matrix2d(size_t r, size_t c, const nng::Vector& cnnv)
{
  Vectord v = cnnv.getVector();
  assert (v.size() == r*c);
  mat.resize(r);
  for (size_t i = 0; i <= v.size() - c ; i += c){
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + c;
    mat[i/c].assign(first, last);
  }
  rows = r;
  cols = c;
}

Matrix2d::Matrix2d(const Matrix2d& rhs) 
{
  mat = rhs.mat;
  rows = rhs.get_rows();
  cols = rhs.get_cols();
}

Matrix2d::~Matrix2d() {}

// Assignment Operator
Matrix2d& Matrix2d::operator=(const Matrix2d& rhs) 
{
  if (&rhs == this)
    return *this;

  size_t new_rows = rhs.get_rows();
  size_t new_cols = rhs.get_cols();

  mat.resize(new_rows);
  for (size_t i=0; i<mat.size(); ++i) 
  {
    mat[i].resize(new_cols);
  }

  for (size_t i=0; i<new_rows; ++i) 
  {
    for (size_t j=0; j<new_cols; ++j) 
	{
      mat[i][j] = rhs(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;

  return *this;
}

// Addition of two matrices
Matrix2d Matrix2d::operator+(const Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (rows == rhs_rows && cols == rhs_cols);

  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] + rhs(i,j);
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
  assert (rows == rhs_rows && cols == rhs_cols);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      mat[i][j] += rhs(i,j);
    }
  }

  return *this;
}


// Subtraction of this matrix and another
Matrix2d Matrix2d::operator-(const Matrix2d& rhs) 
{
  // check dimension
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (rows == rhs_rows && cols == rhs_cols);

  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] - rhs(i,j);
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
  assert (rows == rhs_rows && cols == rhs_cols);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      mat[i][j] -= rhs(i,j);
    }
  }

  return *this;
}


// Left multiplication of this matrix and another
Matrix2d Matrix2d::operator*(const Matrix2d& rhs) 
{
  size_t rhs_rows = rhs.get_rows();
  size_t rhs_cols = rhs.get_cols();
  assert (cols == rhs_rows);

  Matrix2d result(rows, rhs_cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<rhs_cols; ++j) 
	{
      for (size_t k=0; k<cols; k++) 
	  {
        result(i,j) += mat[i][k] * rhs(k,j);
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
  Matrix2d result(rows, cols, 0.0);
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = -mat[i][j];
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
  assert (rows == m_rows && cols == m_cols);

  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] * m(i,j);
    }
  }

  return result;

}

// Calculate a transpose of this matrix

Matrix2d Matrix2d::transpose() 
{
  Matrix2d result(cols, rows, 0.0);

  for (size_t i=0; i<cols; ++i) 
  {
    for (size_t j=0; j<rows; ++j) 
	{
      result(i,j) = mat[j][i];
    }
  }

  return result;
}

const Matrix2d Matrix2d::transpose() const
{
  Matrix2d result(cols, rows, 0.0);

  for (size_t i=0; i<cols; ++i) 
  {
    for (size_t j=0; j<rows; ++j) 
	{
      result(i,j) = mat[j][i];
    }
  }

  return result;
}

/*
Matrix2d& Matrix2d::transpose()
{
  double temp = 0;
  Vectord v = this->toVector();
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=i+1; j<cols; ++j) 
	{
      	temp = v[i*cols+j];
	v[i*cols+j] = v[j*cols+i];
        v[j*cols+i] = temp;
    }
  }
  int r_temp = rows;
  rows = cols;
  cols = r_temp;
  mat.resize(rows);
  for (size_t i = 0; i <= v.size() - cols ; i += cols)
  {
    Vectord::const_iterator first = v.begin() + i;
    Vectord::const_iterator last = v.begin() + i + cols;
    mat[i/cols].assign(first, last);
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
          result(p,q) = mat[i+p][j+q];
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
          mat[i+p][j+q] = m(p,q);
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
      assert (rows == m_rows);

      Matrix2d result(rows, cols + m_cols, 0.0);
      result.setBlock(0,0,rows,cols, *this);
      result.setBlock(0,cols,rows,m_cols, m);

      return result;
  }
  else
  {
      assert (cols == m_cols);

      Matrix2d result(rows + m_rows, cols, 0.0);
      result.setBlock(0,0,rows,cols, *this);
      result.setBlock(rows,0,m_rows,cols, m);

      return result;
  }
}

// sigmoid at each element
Matrix2d Matrix2d::sigmoid()
{
  Matrix2d result(*this);
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = nng::sigmoid(mat[i][j]);
    }
  }
  return result;
}

// sum over rows or columns
nng::Vector Matrix2d::sum(size_t axis)
{
  assert (axis == 0 || axis == 1);
  if (axis == 0) // sum over rows for each column
  {
      nng::Vector result(cols,0);
      for (size_t j=0; j<cols; ++j)
          for (size_t i=0; i<rows; ++i)
              result(j) += mat[i][j];
      return result;
  }
  else // sum over cols for each row
  {
      nng::Vector result(rows,0);
      for (size_t i=0; i<rows; ++i) 
	  {
        for (size_t j=0; j<cols; ++j) 
		{
          result(i) += mat[i][j];
        }
      }
      return result;
  }
}

// sum over entire matrix
double Matrix2d::sum()
{
  double result = 0;
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result += mat[i][j];
    }
  }
  return result;
}

// power at each element
Matrix2d Matrix2d::power(const double exponent)
{
  Matrix2d result(*this);
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = pow(mat[i][j], exponent);
    }
  }
  return result;
}

// derivation of sigmoid at each element
Matrix2d Matrix2d::sigmoid_prime()
{
  Matrix2d result(*this);
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = nng::sigmoid_prime(mat[i][j]);
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

Matrix2d Matrix2d::exp()
{
  Matrix2d result(*this);
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = std::exp(mat[i][j]);
    }
  }
  return result;
}

nng::Vector Matrix2d::argmax(size_t axis)
{
	assert (axis == 0 || axis == 1);
	if (axis == 0) // each element in result is argmax of the column vector of the matrix
	{
		nng::Vector result(cols,0);
		Vectord v;
		size_t arg_max;
		for (size_t i=0; i<cols; ++i)
		{
			v = get_col(i);
			arg_max = std::distance(v.begin(), std::max_element(v.begin(), v.end()));
			result(i) = arg_max;
		}
		return result;
	}
	else
	{
		nng::Vector result(rows,0);
		Vectord v;
		size_t arg_max;
		for (size_t i=0; i<rows; ++i)
		{
			v = mat[i];
			arg_max = std::distance(v.begin(), std::max_element(v.begin(), v.end()));
			result(i) = arg_max;
		}
		return result;
	}
}

Vectord Matrix2d::get_col(size_t index)
{
	Vectord result;
    for (size_t i=0; i<rows; ++i)
	{
		result.push_back(mat[i][index]);
	}
	return result;
}

void Matrix2d::set_col(const Vectord& v, size_t index)
{
	assert (v.size() == rows);
    for (size_t i=0; i<rows; ++i)
	{
		mat[i][index] = v[i];
	}
}

// Matrix/scalar addition
Matrix2d Matrix2d::operator+(const double& rhs) 
{
  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] + rhs;
    }
  }

  return result;
}

// Matrix/scalar subtraction
Matrix2d Matrix2d::operator-(const double& rhs) 
{
  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] - rhs;
    }
  }

  return result;
}


// Matrix/scalar multiplication
Matrix2d Matrix2d::operator*(const double& rhs) 
{
  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] * rhs;
    }
  }

  return result;
}

// Matrix/scalar division
Matrix2d Matrix2d::operator/(const double& rhs) 
{
  assert (std::abs(rhs) > epsilon);

  Matrix2d result(rows, cols, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result(i,j) = mat[i][j] / rhs;
    }
  }

  return result;
}

// Multiply a matrix with a vector
Vectord Matrix2d::operator*(const Vectord& rhs) 
{
  assert (cols == rhs.size());

  Vectord result(rows, 0.0);

  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      result[i] += mat[i][j] * rhs[j];
    }
  }

  return result;
}

// divide each m[i][j] by v[j]
Matrix2d Matrix2d::operator/(const nng::Vector& rhs)
{
	assert (cols == rhs.get_length());
	Matrix2d result(rows, cols, 0.0);
	for (size_t i=0; i<rows; ++i) 
	{
	for (size_t j=0; j<cols; ++j) 
	{
	  result(i,j) = mat[i][j] / rhs[j];
	}
	}

	return result;
}

// Convert from matrix to vector
Vectord Matrix2d::toVector()
{
    Vectord v;
    for (size_t i = 0; i < rows; ++i)
    {
        v.insert(v.end(),mat[i].begin(), mat[i].end());
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
  return mat[row][col];
}

// Access the individual elements (const)
const double& Matrix2d::operator()(const size_t& row, const size_t& col) const 
{
  return mat[row][col];
}


void Matrix2d::print()
{
  for (size_t i=0; i<rows; ++i) 
  {
    for (size_t j=0; j<cols; ++j) 
	{
      std::cout << mat[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}