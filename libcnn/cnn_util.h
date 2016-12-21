#ifndef CNNUTIL_H
#define CNNUTIL_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"

typedef std::vector<double> Vectord;
typedef dlib::matrix<double,0,1> column_vector;

namespace util{
void print_v(const Vectord& v);
double rand_a_b(double a, double b);
double normal_distribution_rand(double mean, double stddev); // generate random number with standard normal distribution
double sigmoid(const double x);
double sigmoid_prime(const double x);
double kl_divergence(const double x, const double y);
}

class  CnnVector
{
    public:
        CnnVector(size_t len, const double& initial);
        CnnVector(const Vectord& v);
        CnnVector(const CnnVector& rhs);
        ~CnnVector(){};

        // Access the individual elements
        double& operator()(const size_t& index);
        const double& operator()(const size_t& index) const;
        double& operator[](const size_t& index);
        const double& operator[](const size_t& index) const;

        // Vector/Vector operations
        CnnVector operator+(const CnnVector& rhs);
        CnnVector operator-(const CnnVector& rhs);
        CnnVector operator/(const CnnVector& rhs);
        CnnVector operator-() const;
		CnnVector dot(const CnnVector& v);

        // Vector/scalar operations
        CnnVector operator+(const double& rhs);
        CnnVector operator-(const double& rhs);
        CnnVector operator*(const double& rhs);
        CnnVector operator/(const double& rhs);

        void print();
        CnnVector getSegment(size_t start_index, size_t len) const;
        void setSegment(size_t start_index, size_t len, const CnnVector& v);
        CnnVector concatenate(const Vectord& v);
        CnnVector concatenate(const CnnVector& v);
        CnnVector kl_divergence(CnnVector& v);
        double sum();


        void setLength(size_t len) {length = len;}
        size_t get_length() const {return length;}
        Vectord& getVector(){return vec;}
        const Vectord& getVector() const {return vec;}

    private:
        Vectord vec;
        size_t length;
};

class Matrix2d
{
public:
    Matrix2d(size_t r, size_t c, const double& initial);
    Matrix2d(size_t r, size_t c, const Vectord& v);
    Matrix2d(size_t r, size_t c, const CnnVector& v);
	Matrix2d(const Matrix2d& rhs);
	~Matrix2d();

	// Matrix/Matrix operations
	Matrix2d& operator=(const Matrix2d& rhs);
	Matrix2d operator+(const Matrix2d& rhs);
  	Matrix2d& operator+=(const Matrix2d& rhs);
  	Matrix2d operator-(const Matrix2d& rhs);
  	Matrix2d& operator-=(const Matrix2d& rhs);
  	Matrix2d operator*(const Matrix2d& rhs);
  	Matrix2d& operator*=(const Matrix2d& rhs);
    Matrix2d operator-() const;
    Matrix2d dot(const Matrix2d& m);

    //Matrix2d dot(const Matrix2d& m);
  	Matrix2d transpose();
  	const Matrix2d transpose() const;
    //Matrix2d& transpose();
    Matrix2d getBlock(size_t i, size_t j, size_t r, size_t c) const;
    void setBlock(size_t i, size_t j, size_t r, size_t c, const Matrix2d& m);
    Matrix2d concatenate(const Matrix2d& m, size_t axis = 0);
    Matrix2d sigmoid();
    CnnVector sum(size_t axis);
    double sum();
    Matrix2d power(const double exponent);
    Matrix2d sigmoid_prime();
	double max();
	Matrix2d exp();
	CnnVector argmax(size_t axis);

  	// Matrix/scalar operations
	Matrix2d operator+(const double& rhs);
	Matrix2d operator-(const double& rhs);
	Matrix2d operator*(const double& rhs);
	Matrix2d operator/(const double& rhs);

  	// Matrix/vector operations
  	Vectord operator*(const Vectord& rhs);
	Matrix2d operator/(const CnnVector& rhs);
    Vectord toVector();
    CnnVector to_cnnvector();

  	// Access the individual elements
  	double& operator()(const size_t& row, const size_t& col);
  	const double& operator()(const size_t& row, const size_t& col) const;

  	// Access the row and column sizes
  	size_t get_rows() const {return rows;}
  	size_t get_cols() const {return cols;}
	Vectord get_col(size_t index = 0);

    void print();

private:
    size_t rows;
    size_t cols;
    std::vector<Vectord> mat;

};

Matrix2d operator+(double scalar, Matrix2d& matrix);
Matrix2d operator-(double scalar, Matrix2d& matrix);
Matrix2d operator*(double scalar, Matrix2d& matrix);
CnnVector operator+(double scalar, CnnVector& v);
CnnVector operator-(double scalar, CnnVector& v);
CnnVector operator*(double scalar, CnnVector& v);

CnnVector column_vector_to_cnn_vector(const column_vector& x);
column_vector cnn_vector_to_column_vector(const CnnVector& v);

#endif
