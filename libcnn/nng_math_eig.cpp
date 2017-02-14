#include "nng_math_eig.h"
#include "nng_math.h"
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
    _v_pk.resize(m.get_cols()-1);
	compute_Pk(m);
}

nng::EigenValueEigenVector::~EigenValueEigenVector()
{
   for (std::vector< nng::Matrix2d* >::iterator it = _v_pk.begin() ; it != _v_pk.end(); ++it)
   {
     delete (*it);
   } 
    
}

std::vector<nng::Matrix2d*> nng::EigenValueEigenVector::compute_Pk(const nng::Matrix2d& A)
{
	size_t n = A.get_cols();
	nng::Vector col_k(n);
	nng::Vector d(n);
	double D;
	nng::Vector v(n);
	double p;
	nng::Matrix2d Pk(n,n);
	nng::Matrix2d m_v(n,1);
	nng::Matrix2d PA(n,n);
    PA = A;
    
	for (size_t k = 0; k < n-1; ++k)
	{
		// pull column k out of matrix P_{k-1}P_{k-2}...P_1A (just A if k=0)
		col_k = nng::Vector(PA.get_col(k));
		// normalize
		d = col_k.normalize();
		//D = +-sqrt(d_k^2 + ... + d_{n-1}^2), choose + if dk<=0
		D = compute_D(d,k);
		// v_0v_1=...=v_{k-1}=0
		v(k) = std::sqrt( 0.5 * ( 1 - d(k)/D) );
		p = -D * v(k);
		for (size_t j = k+1; j <= n; ++j)
		{
			v(j) = d(j)/(2*p);
		}
		// m_v = [0,0,...,0,v_k,v_{k+1},v_{n-1}]^T
		m_v = nng::Matrix2d(n,1,v);
		// P_k = I - 2*m_v*m_v^T
		Pk = nng::Identity(n) - 2.0*m_v*(m_v.transpose());
		// for k+1, PA = P_{k}P_{k-1}...P_1A
        PA = Pk*PA;
		
        _v_pk[k] = new nng::Matrix2d(Pk);
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
