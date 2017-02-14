#ifndef MATRIX2D_H
#define MATRIX2D_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"
#include "Vector.h"

namespace nng{

    class Matrix2d
    {
    public:
        Matrix2d(size_t r, size_t c, const double& initial);
        Matrix2d(size_t r, size_t c, const Vectord& v);
        Matrix2d(size_t r, size_t c, const nng::Vector& v);
        Matrix2d(const Matrix2d& rhs);
        ~Matrix2d();
		

        // Matrix/Matrix operations
        Matrix2d& operator=(const Matrix2d& rhs);
        Matrix2d operator+(const Matrix2d& rhs);
        Matrix2d& operator+=(const Matrix2d& rhs);
        Matrix2d operator-(Matrix2d& rhs);
		Matrix2d operator-(const Matrix2d& rhs) const;
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
        nng::Vector sum(size_t axis);
        double sum();
        Matrix2d power(const double exponent);
        Matrix2d sigmoid_prime();
        double max();
		double min();
        Matrix2d exp();
		Matrix2d log();
		Matrix2d abs();
		Matrix2d floor();
        nng::Vector argmax(size_t axis);

        // Matrix/scalar operations
        Matrix2d operator+(const double& rhs);
        Matrix2d operator-(const double& rhs);
        Matrix2d operator*(const double& rhs);
        Matrix2d operator/(const double& rhs);

        // Matrix/vector operations
        Vectord operator*(const Vectord& rhs);
        Matrix2d operator/(const nng::Vector& rhs);
        Vectord toVector();
        nng::Vector to_cnnvector();

        // Access the individual elements
        double& operator()(const size_t& row, const size_t& col);
        const double& operator()(const size_t& row, const size_t& col) const;

        // Access the row and column sizes
        size_t get_rows() const {return rows;}
        size_t get_cols() const {return cols;}
        Vectord get_col(size_t index = 0);
		Vectord get_row(size_t index = 0) {return mat[index];}
        void set_col(const Vectord& v, size_t index);
		void set_col(const nng::Vector& v, size_t index);
        void setElement(double value, const size_t& row, const size_t& col);
        
        void print();

    private:
        size_t rows;
        size_t cols;
        std::vector<Vectord> mat;

    };
	
	class Identity:public Matrix2d
	{
        public:
        Identity(size_t n);

	};

}

#endif