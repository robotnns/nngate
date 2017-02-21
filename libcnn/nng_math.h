#ifndef NNG_MATH_H
#define NNG_MATH_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"

#include "nng_type.h"
#include "Vector.h"
#include "Matrix2d.h"

namespace nng
{
    
    void print_v(const Vectord& v);
    double rand_a_b(double a, double b);
	size_t rand_a_b(size_t a, size_t b);
    double normal_distribution_rand(double mean, double stddev); // generate random number with standard normal distribution
    double sigmoid(const double x);
    double sigmoid_prime(const double x);
    double kl_divergence(const double x, const double y);



    Vector column_vector_to_cnn_vector(const column_vector& x);
    column_vector cnn_vector_to_column_vector(const Vector& v);
	
	
	typedef std::vector<std::pair<Matrix2d, double>> se_stack; // vector of (w_i,b_i)
	typedef std::vector<std::pair<Matrix2d, Vector>> se_stack_grad; // vector of (grad_w_i, grad_b_i)

	struct se_net_config
	{
		size_t input_size;
		std::vector<size_t> layer_sizes;
	};

	struct param_config
	{
		nng::Vector params; // vector of w_i b_i
		se_net_config  net_config;
		param_config(nng::Vector& params, se_net_config& net_config):params(params), net_config(net_config){};
	};

	param_config stack2params(se_stack& stack);
	param_config stack2params(se_stack_grad& stack_grad);
	se_stack params2stack(param_config& param_net_config);
	

    
    nng::Matrix2d operator+(double scalar, nng::Matrix2d& matrix);
    nng::Matrix2d operator-(double scalar, nng::Matrix2d& matrix);
    nng::Matrix2d operator*(double scalar, nng::Matrix2d& matrix);
    nng::Vector operator+(double scalar, nng::Vector& v);
    nng::Vector operator-(double scalar, nng::Vector& v);
    nng::Vector operator*(double scalar, nng::Vector& v);
    nng::Vector operator/(double scalar, nng::Vector& v);
}
#endif
