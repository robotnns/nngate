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
            
            pair_eigenvalue_eigenvector compute_eigenvalue_eigenvector_QR(Matrix2d& A);
            bool compute_QR(Matrix2d& A);
            void compute_Pk(const Matrix2d& A); // compute householder matrices
            double compute_D(const Vector& d, size_t k);
            
        private:
            std::vector<Matrix2d*> _v_pk;
            nng::Matrix2d _A;
    };
}

#endif
