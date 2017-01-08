#include "StackedAutoencoder.h"
#include <math.h>
#include <iostream>


nng::StackedAutoencoder::StackedAutoencoder(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data):
    visible_size(visible_size)
    ,hidden_size(hidden_size)
    ,sparsity_param(sparsity_param)
    ,lambda(lambda)
    ,beta(beta)
    ,grad(nng::Vector(hidden_size * visible_size * 2 + hidden_size + visible_size, 0))
    ,data(data)
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

nng::se_stack StackedAutoencoder::params2stack(param_config& param_net_config)
{
	
}