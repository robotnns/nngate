#include "Matrix4d.h"
#include <iostream>
#include <assert.h>
#include <math.h>


using namespace nng;
using namespace std;

Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4) 
{
  _mat.resize(s1);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
	for (size_t j = 0; j < s2; ++j)
	{
	  _mat[i].push_back(Matrix2d(s3, s4));
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
	for (size_t j = 0; j < s2; ++j)
	{
	  _mat[i].push_back(Matrix2d(s3, s4, initial));
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
  _mat.resize(s1);
  Matrix2d m(s3, s4);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
	for (size_t j = 0; j < s2; ++j)
	{
	  m = Matrix2d(s3,s4,v.getSegment(i*s2*s3*s4 + j*s3*s4,s3*s4));
	  _mat[i].push_back(m);
	}
  }

  _shape.resize(4);
  _shape[0] = s1;
  _shape[1] = s2;
  _shape[2] = s3;
  _shape[3] = s4;
}

Matrix4d::Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const nng::Vector& v)
{
  assert (v.get_length() == s1*s2*s3*s4);
  _mat.resize(s1);
  Matrix2d m(s3, s4);
  for (size_t i=0; i<_mat.size(); ++i) 
  {
	for (size_t j = 0; j < s2; ++j)
	{
	  m = Matrix2d(s3,s4,v.getSegment(i*s2*s3*s4 + j*s3*s4,s3*s4));
	  _mat[i].push_back(m);
	}
  }

  _shape.resize(4);
  _shape[0] = s1;
  _shape[1] = s2;
  _shape[2] = s3;
  _shape[3] = s4;
}

Matrix4d::Matrix4d(const Matrix4d& rhs) 
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


Matrix4d::~Matrix4d() {}



// Access the individual elements
double& Matrix4d::operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4) 
{
  return _mat[s1][s2](s3,s4);
}

// Access the individual elements (const)
const double& Matrix4d::operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4)  const 
{
  return _mat[s1][s2](s3,s4);
}


void Matrix4d::print()
{
  for (size_t i=0; i<_shape[0]; ++i) 
  {
    for (size_t j=0; j<_shape[1]; ++j) 
	{
		std::cout << "i="<<i << ", j="<<j<<std::endl;
		for (size_t k=0; k<_shape[2]; ++k) 
		{
			for (size_t l=0; l<_shape[3]; ++l) 
			{
			  std::cout << _mat[i][j](k,l) << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

