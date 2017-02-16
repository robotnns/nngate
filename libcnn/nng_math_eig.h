#ifndef NNG_MATH_EIG_H
#define NNG_MATH_EIG_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"

#include "Vector.h"
#include "Matrix2d.h"

namespace nng
{
    typedef std::pair<Vector, Matrix2d> pair_eigenvalue_eigenvector;

    class EigenValueEigenVector
    {
        public:
            EigenValueEigenVector(const Matrix2d& A);
            ~EigenValueEigenVector();
            
            void compute_eigenvalue_eigenvector_QR();
            void compute_QR(Matrix2d& A);
            void compute_Pk(const Matrix2d& A); // compute householder matrices
            double compute_D(const Vector& d, size_t k);
            
			const Vector& getEigenValue() const {return _eig.first;};
			const Matrix2d& getEigenVector() const {return _eig.second;};
        private:
            std::vector<Matrix2d*> _v_pk;
            nng::Matrix2d _A;
			nng::Matrix2d _S;
			nng::pair_eigenvalue_eigenvector _eig;
    };
}

#endif
