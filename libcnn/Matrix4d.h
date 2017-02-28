#ifndef MATRIX4D_H
#define MATRIX4D_H

#include <vector>
#include <cmath>
#include <stdlib.h>

#include "nng_type.h"
#include "Vector.h"

namespace nng{

	class Vector;
	
    class Matrix4d
    {
    public:
		Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4);
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const double& initial);
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const Vectord& v);
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const nng::Vector& v);
        Matrix4d(const Matrix4d& rhs);
		Matrix4d(Matrix4d&& rhs);
		Matrix4d& operator=(Matrix4d&& rhs);
        ~Matrix4d();
		

        // Access the individual elements
        double& operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4);
        const double& operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4) const;

        // Access the row and column sizes
		std::vector<size_t> shape const {return _shape;}
        Vectord get_col(size_t index = 0);
		Vectord get_row(size_t index = 0) {return _mat[index];}
        void set_col(const Vectord& v, size_t index);
		void set_col(const nng::Vector& v, size_t index);
        void setElement(double value, const size_t& row, const size_t& col);
        
        void print();

    private:
        std::vector<size_t> _shape;
        std::vector<std::vector<std::vector<Vectord>>> _mat;

    };
	
	
#endif