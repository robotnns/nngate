#include "StackedAutoencoder.h"
#include <math.h>
#include <iostream>


nng::StackedAutoencoder::StackedAutoencoder(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, double beta, size_t m, nng::Matrix2d& data, nng::Vector& labels;):
    input_size(input_size)
    ,hidden_size(hidden_size)
    ,num_classes(num_classes)
    ,net_config(net_config)
    ,lambda(lambda)
    ,data(data)
    ,labels(labels)
{
}

nng::StackedAutoencoder::~StackedAutoencoder()
{
}

//typedef se_net_config std::map<std::string,std::vector<size_t>>;	
//typedef se_stack std::vector<std::pair<Matrix2d, double>>;
nng::param_config StackedAutoencoder::stack2params(se_stack& stack)
{
    nng::Vectord params(0,0.0); 
	
    for (std::pair s : stack)
	{
		nng::Matrix2d w(s.first);
        params = params.concatenate(w.toVector());//s['w']
		nng::Vectord b(1,s.second);
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
        for (std::pair s : stack)
		{
            net_config.layer_sizes.push_back(s.first.get_rows());
		}
	}	
	nng::param_config param_net_config;
	param_net_config.params = params;
	param_net_config.net_config = net_config;
	
	return param_net_config;
}

//Converts a flattened parameter vector into a "stack" structure for multilayer networks
nng::se_stack StackedAutoencoder::params2stack(param_config& param_net_config)
{
    nng::se_net_config net_config = param_net_config.net_config;
    Vectord params = param_net_config.params;
    
    // Map the params (a vector into a stack of weights)
    size_t depth = net_config.layer_sizes.size();
    se_stack stack;
    //stack = [dict() for i in range(depth)]

    size_t prev_layer_size = net_config.input_size;
    size_t current_pos = 0
    size_t current_layer_size = 0;
    
    for (size_t i = 0; i < depth; i++)
    {
        current_layer_size = net_config.layer_sizes.at(i); 
        // Extract weights and bias
        size_t wlen = prev_layer_size * current_layer_size; 
        //stack[i]['w'] = params[current_pos:current_pos + wlen].reshape(net_config['layer_sizes'][i], prev_layer_size)
        //stack[i]['b'] = params[current_pos:current_pos + blen]
        stack.push_back(std::pair(nng::Matrix2d(current_layer_size, prev_layer_size, params.getSegment(current_pos, wlen)),
                                  params(current_pos + wlen)));

        current_pos = current_pos + wlen + 1;

        // Set previous layer size
        prev_layer_size = current_layer_size;
    }

    return stack
}