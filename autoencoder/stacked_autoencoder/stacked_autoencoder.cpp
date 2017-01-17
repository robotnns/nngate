#include "stacked_autoencoder.h"
#include <math.h>
#include <iostream>


nng::StackedAutoencoder::StackedAutoencoder(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels):
    _input_size(input_size)
    ,_hidden_size(hidden_size)
    ,_num_classes(num_classes)
    ,_net_config(net_config)
    ,_lambda(lambda)
    ,_data(data)
    ,_labels(labels)
{
}

nng::StackedAutoencoder::~StackedAutoencoder()
{
}

//typedef se_net_config std::map<std::string,std::vector<size_t>>;	
//typedef se_stack std::vector<std::pair<Matrix2d, double>>;
nng::param_config nng::stack2params(se_stack& stack)
{
    nng::Vector params(0,0.0); 
	
    for (std::pair<nng::Matrix2d, double> s : stack)
	{
		nng::Matrix2d w(s.first);
        params = params.concatenate(w.toVector());//s['w']
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
        params = params.concatenate(w.toVector());//s['w']
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

double nng::StackedAutoencoder::operator() (nng::column_vector x) const
{
	return compute_cost(x);
}

double nng::StackedAutoencoder::compute_cost(nng::column_vector& x) const
{
    nng::Vector v = nng::column_vector_to_cnn_vector(x);
    return do_compute_cost(v);	
}

double nng::StackedAutoencoder::do_compute_cost(nng::Vector& theta) const
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

	double cost =  (-1.0 / m) *( (indicator.dot(prob.log())).sum() ) + 0.5 * _lambda * (softmax_theta.dot(softmax_theta)).sum();
	
	return cost;
}