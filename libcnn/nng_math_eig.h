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
    
	pair_eigen_value_eigenvector compute_eigenvalue_eigenvector_QR(const Matrix2d& m);
	
	Matrix2d compute_Pk(const Matrix2d& m);
	double compute_D(const Vector& d, size_t k);
}

#endif
