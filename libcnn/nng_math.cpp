#include "nng_math.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <random>

const double epsilon = 0.000001;

void nng::print_v(const Vectord& v)
{
  for (size_t i=0; i<v.size(); i++)
    std::cout<<v[i]<<" ";
  std::cout<<std::endl;
}

double nng::rand_a_b(double a, double b)
{
    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}

size_t nng::rand_a_b(size_t a, size_t b)
{
    return (( rand()/(size_t)RAND_MAX ) * (b-a) + a);
}

double nng::normal_distribution_rand(double mean, double stddev)
{
    std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, stddev);
	return distribution(generator);
}

double nng::sigmoid(const double x)
{
  return 1 / (1 + exp(-x));
}

double nng::sigmoid_prime(const double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

double nng::kl_divergence(const double x, const double y)
{
  return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y));
}

nng::Matrix2d nng::operator+(double scalar, nng::Matrix2d& matrix) 
{
    return matrix + scalar;
}
nng::Matrix2d nng::operator-(double scalar, nng::Matrix2d& matrix) 
{
    size_t rows = matrix.get_rows();
    size_t cols = matrix.get_cols();
    nng::Matrix2d m_scalar(rows,cols,scalar);
    return m_scalar - matrix;
}

nng::Matrix2d nng::operator*(double scalar, nng::Matrix2d& matrix) 
{
    return matrix * scalar;
}

nng::Vector nng::operator+(double scalar, nng::Vector& v) 
{
    return v + scalar;
}
nng::Vector nng::operator-(double scalar, nng::Vector& v) 
{
    size_t len = v.get_length();
    nng::Vector v_scalar(len,scalar);
    return v_scalar - v;
}

nng::Vector nng::operator*(double scalar, nng::Vector& v) 
{
    return v * scalar;
}

nng::Vector nng::operator/(double scalar, nng::Vector& v) 
{
    size_t len = v.get_length();
    nng::Vector v_scalar(len,scalar);
    return v_scalar / v;
}

nng::Vector nng::column_vector_to_cnn_vector(const nng::column_vector& x)
{
    size_t len = x.nr();
    nng::Vector v(len,0);
    for(size_t i = 0; i < len; ++i)
    {
        v(i) = x(i);
    }	
	return v;
}

nng::column_vector nng::cnn_vector_to_column_vector(const nng::Vector& v)
{
	size_t len = v.get_length();
    nng::column_vector x(len);
    for(size_t i = 0; i < len; ++i)
    {
        x(i) = v(i);
    }
    return x;
}

//typedef se_net_config std::map<std::string,std::vector<size_t>>;	
//typedef se_stack std::vector<std::pair<Matrix2d, double>>;
nng::param_config nng::stack2params(se_stack& stack)
{
    nng::Vector params(0,0.0); 
	
    for (std::pair<nng::Matrix2d, double> s : stack)
	{
		nng::Matrix2d w(s.first);
        params = params.concatenate(w.to_cnnvector());//s['w']
		nng::Vector b(1,s.second);
        params = params.concatenate(b);//s['b']
	}

    nng::se_net_config net_config;

    if (stack.size() == 0)
	{
        net_config.input_size = 0;
	}
    else
	{
        net_config.input_size = stack[0].first.get_cols();
        for (std::pair<nng::Matrix2d, double> s : stack)
		{
            net_config.layer_sizes.push_back(s.first.get_rows());
		}
	}	
	nng::param_config param_net_config(params,net_config);

	
	return param_net_config;
}

nng::param_config nng::stack2params(se_stack_grad& stack_grad)
{
    nng::Vector params(0,0.0); 
	
    for (std::pair<nng::Matrix2d, nng::Vector> s : stack_grad)
	{
		nng::Matrix2d w(s.first);
        params = params.concatenate(w.to_cnnvector());//s['w']
		nng::Vector b(s.second);
        params = params.concatenate(b);//s['b']
	}

    nng::se_net_config net_config;

    if (stack_grad.size() == 0)
	{
        net_config.input_size = 0;
	}
    else
	{
        net_config.input_size = stack_grad[0].first.get_cols();
        for (std::pair<nng::Matrix2d, nng::Vector> s : stack_grad)
		{
            net_config.layer_sizes.push_back(s.first.get_rows());
		}
	}	
	nng::param_config param_net_config(params,net_config);

	
	return param_net_config;
}

//Converts a flattened parameter vector into a "stack" structure for multilayer networks
nng::se_stack nng::params2stack(nng::param_config& param_net_config)
{
    nng::se_net_config net_config = param_net_config.net_config;
    nng::Vector params = param_net_config.params;
    
    // Map the params (a vector into a stack of weights)
    size_t depth = net_config.layer_sizes.size();
    nng::se_stack stack;
    //stack = [dict() for i in range(depth)]

    size_t prev_layer_size = net_config.input_size;
    size_t current_pos = 0;
    size_t current_layer_size = 0;
    
    for (size_t i = 0; i < depth; i++)
    {
        current_layer_size = net_config.layer_sizes.at(i); 
        // Extract weights and bias
        size_t wlen = prev_layer_size * current_layer_size; 
        //stack[i]['w'] = params[current_pos:current_pos + wlen].reshape(net_config['layer_sizes'][i], prev_layer_size)
        //stack[i]['b'] = params[current_pos:current_pos + blen]
        stack.push_back(std::pair<nng::Matrix2d, double>(nng::Matrix2d(current_layer_size, prev_layer_size, params.getSegment(current_pos, wlen)),
                                  params(current_pos + wlen)));

        current_pos = current_pos + wlen + 1;

        // Set previous layer size
        prev_layer_size = current_layer_size;
    }

    return stack;
}

nng::Matrix2d nng::convolve2d(const nng::Matrix2d& im, const nng::Matrix2d& feature,const size_t stride)
{
	size_t im_width = im.get_cols();
	size_t im_height = im.get_rows();
	size_t feature_width = feature.get_cols();
	size_t feature_height = feature.get_rows();
	
    assert((im_width - feature_width)%stride == 0);
	assert((im_height - feature_height)%stride == 0);
	
	size_t output_width = (im_width - feature_width)/stride + 1;
	size_t output_height = (im_height - feature_height)/stride + 1;
	nng::Matrix2d result(output_width, output_height);
	

	//if ((Wx-Wf)%s != 0 || (Hx-Hf)%s != 0)
	//{
	//	std::cout<<"error: stride value not compatible with data and filter size"<<std::endl;
	//	return false;
	//}
	
	for (size_t i=0;i<=(im_width-feature_width);i+=stride)
	{
		for (size_t j=0;j<=(im_height-feature_height);j+=stride)
		{
			
			result(i,j) = im.getBlock(i,j,feature_width,feature_height).dot(feature).sum();
			//double sum = 0;
			//for (size_t l=0;l<feature_width;l++)
			//{
			//	for (size_t k=0;k<feature_height;k++)
			//	{	
			//		sum += X[Wx*(i+l)+j+k]*F[Wf*l+k];
			//	}
			//}
			//output.push_back(sum);
		
		}
	}
	return result;
}