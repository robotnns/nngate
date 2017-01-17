#include "StackedAutoencoderGrad.h"
#include <math.h>
#include <iostream>


nng::StackedAutoencoderGrad::StackedAutoencoderGrad(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels):
    _input_size(input_size)
    ,_hidden_size(hidden_size)
    ,_num_classes(num_classes)
    ,_net_config(net_config)
    ,_lambda(lambda)
    ,_data(data)
    ,_labels(labels)
{
}

nng::StackedAutoencoderGrad::~StackedAutoencoderGrad()
{
}

const nng::column_vector nng::StackedAutoencoderGrad::operator() (const nng::column_vector x) const {  return compute_grad(x); } 



const nng::column_vector nng::StackedAutoencoderGrad::compute_grad(const nng::column_vector& x) const
{

    nng::Vector v = nng::column_vector_to_cnn_vector(x);
    nng::Vector vgrad = do_compute_grad(v);
    nng::column_vector result = nng::cnn_vector_to_column_vector(vgrad);
    return result;
}
        

const nng::Vector nng::StackedAutoencoderGrad::do_compute_grad(nng::Vector& theta) const
{
    // extract the part which compute the softmax gradient
    nng::Matrix2d softmax_theta(_num_classes, _hidden_size, theta.getSegment(0,_hidden_size*_num_classes));

    // Extract out the "stack"
	nng::Vector params = theta.getSegment(_hidden_size*_num_classes,theta.get_length() - _hidden_size*_num_classes);
	nng::se_net_config net_config = _net_config;
	nng::param_config param_net_config(params,net_config);
    nng::se_stack stack = params2stack(param_net_config);

    size_t m = _data.get_cols();

    // Forward propagation
    std::vector<nng::Matrix2d> a;
	a.push_back(_data);
    std::vector<nng::Matrix2d> z; 
	z.push_back(nng::Matrix2d(1,1,0)); // Dummy value

    for (std::pair<nng::Matrix2d, double> s : stack)
	{
        z.push_back(s.first*(a.back()) + s.second);
        a.push_back(z.back().sigmoid());
	}
    // Softmax
    nng::Matrix2d prod = softmax_theta*(a.back());
    prod = prod - prod.max();
	
    nng::Matrix2d prob = prod.exp() / prod.exp().sum(0);
	nng::Matrix2d indicator(_num_classes, m, 0.0); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(_labels(i), i) = 1.0; // the class of the i-th example is labels(i)
	}
	
	nng::Matrix2d diff_indicator_prob = indicator - prob;
    nng::Matrix2d softmax_grad = (-1.0 / m) * diff_indicator_prob * (a.back().transpose()) + _lambda * softmax_theta;
	

    // Backprop
    // Compute partial of cost (J) w.r.t to outputs of last layer (before softmax)
    nng::Matrix2d softmax_grad_a = softmax_theta.transpose()*diff_indicator_prob;

    // Compute deltas
	std::vector<nng::Matrix2d> delta;
	delta.push_back(-softmax_grad_a.dot(z.back().sigmoid_prime()));
    //delta = [-softmax_grad_a * sigmoid_prime(z[-1])]
	nng::Matrix2d d(0,0,0);
	std::vector<nng::Matrix2d>::iterator it;
	for (size_t i = stack.size()-1; i>=0; i--)
	{
		d = stack[i].first.transpose()*delta[0];
		d = d.dot(z[i].sigmoid_prime());
		it = delta.begin();
		delta.insert(it,d);
	}
    //for i in reversed(range(len(stack))):
      //  d = stack[i]['w'].transpose().dot(delta[0]) * sigmoid_prime(z[i])
      //  delta.insert(0, d)

    // Compute gradients
	nng::se_stack_grad stack_grad;
	nng::Matrix2d grad_w_i(0,0,0);
	nng::Vector grad_b_i(0,0);
	for (size_t i = 0; i<stack.size(); i++)
	{
		grad_w_i = (delta[i+1]*a[i].transpose())/(double)m;
		grad_b_i = delta[i+1].sum(1)/(double)m;
		
		stack_grad.push_back(std::pair<nng::Matrix2d, nng::Vector>(grad_w_i,grad_b_i));
	}

	nng::param_config param_net_config_grad = nng::stack2params(stack_grad);
	nng::Vector grad_params = param_net_config_grad.params;
	nng::Vector grad = softmax_grad.to_cnnvector().concatenate(grad_params);
 
    return grad;
}