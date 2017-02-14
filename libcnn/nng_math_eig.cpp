#include "nng_math_eig.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

const double epsilon = 0.000001;

//nng::pair_eigen_value_eigenvector nng::compute_eigenvalue_eigenvector_QR(const Matrix2d& m)

nng::EigenValueEigenVector::EigenValueEigenVector(const nng::Matrix2d& m)
{
    _v_pk.resize(m.get_cols());
}

nng::EigenValueEigenVector::~EigenValueEigenVector()
{
    for (size_t i = 0; i < _v_pk.size(); ++i)
        delete[] _v_pk[i];
    
}

std::vector<nng::Matrix2d*> nng::EigenValueEigenVector::compute_Pk(const nng::Matrix2d& m)
{
	size_t n = m.get_cols();
    nng::Vectord col_k_tmp;
	nng::Vector col_k(n,0.0);
	nng::Vector d(n,0.0);
	double D;
	nng::Vector v(n,0.0);
	double p;
	nng::Matrix2d Pk(n,n,0.0);
	nng::Matrix2d m_v(n,1,0.0);
	nng::Matrix2d PA(n,n,0.0);
    PA = m;
    
	for (size_t k = 0; k < n; ++k)
	{
        col_k_tmp = PA.get_col(k);
		col_k = nng::Vector(col_k_tmp);
		d = col_k.normalize();
		D = compute_D(d,k);
		v(k) = std::sqrt( 0.5 * ( 1 - d(k)/D) );
		p = -D * v(k);
		for (size_t j = k+1; j <= n; ++j)
		{
			v(j) = d(j)/(2*p);
		}
		m_v = nng::Matrix2d(n,1,v);
		Pk = nng::Identity(n);// - 2.0*m_v*(m_v.transpose());
        PA = Pk*PA;
        _v_pk.push_back(new nng::Matrix2d(Pk));
	}
    return _v_pk;
}

//d=[d1,d2,...,dk,...dn]
//return D = +-sqrt(dk^2 + ... + dn^2), choose + if dk<=0
double nng::EigenValueEigenVector::compute_D(const Vector& d, size_t k)
{
	assert (k > 0 && k <= d.get_length());
	double result = d.getTail(k-1).norm2();
	if (d(k-1) > 0)
		result = -result;
	return result;
}
