#include "StackedAutoencoder.h"
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
    //std::cout << "StackedAutoencoder: extract the part which compute the softmax gradient" << std::endl;
    nng::Matrix2d softmax_theta(_num_classes, _hidden_size, theta.getSegment(0,_hidden_size*_num_classes));

    //std::cout << "StackedAutoencoder: Extract out the stack" << std::endl;
	nng::Vector params = theta.getSegment(_hidden_size*_num_classes,theta.get_length() - _hidden_size*_num_classes);
	nng::se_net_config net_config = _net_config;
	nng::param_config param_net_config(params,net_config);
    nng::se_stack stack = nng::params2stack(param_net_config);

    size_t m = _data.get_cols();

    //std::cout << "StackedAutoencoder: Forward propagation" << std::endl;
    std::vector<nng::Matrix2d> a;
	a.push_back(_data);
    std::vector<nng::Matrix2d> z; 
	z.push_back(nng::Matrix2d(0,0)); // Dummy value

    for (std::pair<nng::Matrix2d, double> s : stack)
	{
        z.push_back(s.first*(a.back()) + s.second);
        a.push_back(z.back().sigmoid());
	}
    //std::cout << "StackedAutoencoder: Softmax" << std::endl;
    nng::Matrix2d prod = softmax_theta*(a.back());
    prod = prod - prod.max();
	
    nng::Matrix2d prob = prod.exp() / prod.exp().sum(0);
	nng::Matrix2d indicator(_num_classes, m); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(_labels(i), i) = 1.0; // the class of the i-th example is labels(i)
	}

	double cost =  (-1.0 / m) *( (indicator.dot(prob.log())).sum() ) + 0.5 * _lambda * (softmax_theta.dot(softmax_theta)).sum();
	std::cout << "StackedAutoencoder: cost = " << cost << std::endl;
	return cost;
}

nng::Vector nng::StackedAutoencoder::stacked_autoencoder_predict(nng::Vector& opt_theta, nng::Matrix2d& data, nng::Softmax& sm)
{
	//std::cout << "StackedAutoencoder::stacked_autoencoder_predict " << std::endl;
// Takes a trained theta and a test data set, and returns the predicted labels for each example
//    :param theta: trained weights from the autoencoder
//    :param input_size: the number of input units
//    :param hidden_size: the number of hidden units at the layer before softmax
//    :param num_classes: the number of categories
//    :param netconfig: network configuration of the stack
//    :param data: the matrix containing the training data as columsn. data[:,i-1] is the i-th training example
//    :return: the prediction matrix pred, where pred(i) is argmax_c P(y(c) | x(i)).


    // Unroll theta parameter
	// extract the part which compute the softmax gradient
    nng::Matrix2d softmax_theta(_num_classes, _hidden_size, opt_theta.getSegment(0,_hidden_size * _num_classes));

    // Extract out the "stack"
	nng::Vector params = opt_theta.getSegment(_hidden_size * _num_classes,opt_theta.get_length() - _hidden_size * _num_classes);
	nng::se_net_config net_config = _net_config;
	nng::param_config param_net_config(params, net_config);
    nng::se_stack stack = nng::params2stack(param_net_config);

    //size_t m = data.get_cols();

    // Compute predictions
    std::vector<nng::Matrix2d> a; a.push_back(data);
    std::vector<nng::Matrix2d> z; 
	z.push_back(nng::Matrix2d(0,0)); // Dummy value

	//Sparse Autoencoder Computation
    for (std::pair<nng::Matrix2d, double> s : stack)
	{
        z.push_back(s.first*(a.back()) + s.second);
        a.push_back(z.back().sigmoid());
	}

    // Softmax
	nng::Vector sm_theta = softmax_theta.to_cnnvector();
    nng::Vector pred = sm.softmax_predict(sm_theta, a.back());

    return pred;
}