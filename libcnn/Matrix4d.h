#ifndef MATRIX4D_H
#define MATRIX4D_H

#include <vector>
#include <cmath>
#include <stdlib.h>

#include "nng_type.h"
#include "Vector.h"
#include "Matrix2d.h"

namespace nng{
	
    class Matrix4d
    {
    public:
		Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4);//(r, c, channel, image number)
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const double& initial);
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const Vectord& vec);
        Matrix4d(size_t s1, size_t s2, size_t s3, size_t s4, const nng::Vector& v);
        Matrix4d(const Matrix4d& rhs);
		Matrix4d(Matrix4d&& rhs);
		Matrix4d& operator=(Matrix4d&& rhs);
		Matrix4d& operator=(const Matrix4d& rhs);
        ~Matrix4d();
		

        // Access the individual elements
        double& operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4);
        const double& operator()(const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4) const;


		std::vector<size_t> shape() const {return _shape;}
		Matrix2d getMatrix2d(const size_t& s3, const size_t& s4){return _mat[s4][s3];}
		void setMatrix2d(const nng::Matrix2d& m, const size_t& s3, const size_t& s4){_mat[s4][s3] = m;}
        //void setElement(double value, const size_t& s1, const size_t& s2, const size_t& s3, const size_t& s4);
        
       void print();

    private:
        std::vector<size_t> _shape;
        std::vector<std::vector<Matrix2d>> _mat;

    };
	
}	
#endif