#include "cnn_util.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

const double epsilon = 0.000001;

void util::print_v(const Vectord& v)
{
  for (size_t i=0; i<v.size(); i++)
    std::cout<<v[i]<<" ";
  std::cout<<std::endl;
}

double util::rand_a_b(double a, double b)
{
    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}

double util::normal_distribution_rand(double mean, double stddev)
{
    std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, stddev);
	return distribution(generator);
}

double util::sigmoid(const double x)
{
  return 1 / (1 + exp(-x));
}

double util::sigmoid_prime(const double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

double util::kl_divergence(const double x, const double y)
{
  return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y));
}



CnnVector::CnnVector(size_t len, const double& initial)
{
    vec.resize(len);
    for (size_t i=0; i<len; ++i)
        vec[i] = initial;
    length = len;
}


CnnVector::CnnVector(const Vectord& v):
vec(v)
{
    length = v.size();
}

CnnVector::CnnVector(const CnnVector& rhs) {
  vec = rhs.vec;
  length = rhs.get_length();
}

// Access the individual elements
double& CnnVector::operator()(const size_t& index)
{
    return vec[index];
}

const double& CnnVector::operator()(const size_t& index) const
{
    return vec[index];
}

double& CnnVector::operator[](const size_t& index)
{
    return vec[index];
}

const double& CnnVector::operator[](const size_t& index) const
{
    return vec[index];
}

//Vector/Vector addition
CnnVector CnnVector::operator+(const CnnVector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] + rhs(i);
  }

  return result;

}

//Vector/Vector subtraction
CnnVector CnnVector::operator-(const CnnVector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] - rhs(i);
  }

  return result;

}

//Vector/Vector division
CnnVector CnnVector::operator/(const CnnVector& rhs)
{
  // check dimension
  assert (length == rhs.get_length());

  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] / rhs(i);
  }

  return result;

}

// Vector/Vector dot product
CnnVector CnnVector::dot(const CnnVector& v)
{
  // check dimension
  size_t len = v.get_length();
  assert (length == len);

  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) 
  {
      result(i) = vec[i]* v(i);
  }

  return result;

}

// Vector/scalar addition
CnnVector CnnVector::operator+(const double& rhs) 
{
  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] + rhs;
  }

  return result;
}

// Vector/scalar subtraction
CnnVector CnnVector::operator-(const double& rhs) 
{
  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] - rhs;
  }

  return result;
}

// unary - operator
CnnVector CnnVector::operator-() const
{
  CnnVector result(length, 0.0);
  for (size_t i=0; i<length; ++i) {
      result(i) = -vec[i];
  }

  return result;
}

// Vector/scalar multiplication
CnnVector CnnVector::operator*(const double& rhs) 
{
  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] * rhs;
  }

  return result;
}
// Vector/scalar division
CnnVector CnnVector::operator/(const double& rhs) 
{
  assert (std::abs(rhs) > epsilon);

  CnnVector result(length, 0.0);

  for (size_t i=0; i<length; ++i) {
      result(i) = vec[i] / rhs;
  }

  return result;
}

void CnnVector::print()
{
  for (size_t i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl;
}

// Get submatrix of length = len starting at start_index
CnnVector CnnVector::getSegment(size_t start_index, size_t len) const
{
  assert (start_index < length && start_index + len < length);
  CnnVector result(len, 0.0);

  for (size_t i=0; i<len; ++i) {
      result(i) = vec[start_index+i];
  }

  return result;
}

void CnnVector::setSegment(size_t start_index, size_t len, const CnnVector& v)
{
  for (size_t i=0; i<len; ++i) {
      vec[start_index+i] = v(i);
  }
}


CnnVector CnnVector::concatenate(const Vectord& v)
{
    CnnVector result(*this);
    result.getVector().insert(result.getVector().end(),v.begin(), v.end());
    result.setLength(length + v.size());
    return result;
}

CnnVector CnnVector::concatenate(const CnnVector& v)
{
    CnnVector result(*this);
    result.getVector().insert(result.getVector().end(),v.getVector().begin(), v.getVector().end());
    result.setLength(length + v.get_length());
    return result;
}

// kl divergence between each element in the current vector and v
CnnVector CnnVector::kl_divergence(CnnVector& v)
{
    assert (length == v.get_length());
    CnnVector result(*this);
    double sum_vec = sum();
    double sum_v = v.sum();

    for (size_t i=0; i<length; ++i) {
        result(i) = util::kl_divergence(vec[i]/sum_vec, v(i)/sum_v);
    }
    return result;
}

// sum of all the elements in the vector
double CnnVector::sum()
{
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}

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

Matrix2d::Matrix2d(size_t r, size_t c, const CnnVector& cnnv)
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
      result(i,j) = util::sigmoid(mat[i][j]);
    }
  }
  return result;
}

// sum over rows or columns
CnnVector Matrix2d::sum(size_t axis)
{
  assert (axis == 0 || axis == 1);
  if (axis == 0) // sum over rows for each column
  {
      CnnVector result(cols,0);
      for (size_t j=0; j<cols; ++j)
          for (size_t i=0; i<rows; ++i)
              result(j) += mat[i][j];
      return result;
  }
  else // sum over cols for each row
  {
      CnnVector result(rows,0);
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
      result(i,j) = util::sigmoid_prime(mat[i][j]);
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

CnnVector Matrix2d::argmax(size_t axis)
{
	assert (axis == 0 || axis == 1);
	if (axis == 0) // each element in result is argmax of the column vector of the matrix
	{
		CnnVector result(cols,0);
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
		CnnVector result(rows,0);
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
Matrix2d Matrix2d::operator/(const CnnVector& rhs)
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

CnnVector Matrix2d::to_cnnvector()
{
    return CnnVector(this->toVector());
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

Matrix2d operator+(double scalar, Matrix2d& matrix) 
{
    return matrix + scalar;
}
Matrix2d operator-(double scalar, Matrix2d& matrix) 
{
    size_t rows = matrix.get_rows();
    size_t cols = matrix.get_cols();
    Matrix2d m_scalar(rows,cols,scalar);
    return m_scalar - matrix;
}

Matrix2d operator*(double scalar, Matrix2d& matrix) 
{
    return matrix * scalar;
}

CnnVector operator+(double scalar, CnnVector& v) 
{
    return v + scalar;
}
CnnVector operator-(double scalar, CnnVector& v) 
{
    size_t len = v.get_length();
    CnnVector v_scalar(len,scalar);
    return v_scalar - v;
}

CnnVector operator*(double scalar, CnnVector& v) 
{
    return v * scalar;
}

CnnVector column_vector_to_cnn_vector(const column_vector& x)
{
    size_t len = x.nr();
    CnnVector v(len,0);
    for(size_t i = 0; i < len; ++i)
    {
        v(i) = x(i);
    }	
	return v;
}

column_vector cnn_vector_to_column_vector(const CnnVector& v)
{
	size_t len = v.get_length();
    column_vector x(len);
    for(size_t i = 0; i < len; ++i)
    {
        x(i) = v(i);
    }
    return x;
}