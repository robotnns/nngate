#ifndef STACKED_AUTOENCODER_H
#define STACKED_AUTOENCODER_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{

	
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
			
    class StackedAutoencoder
    {
        public:
            StackedAutoencoder(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels);
            ~StackedAutoencoder();


            double operator() (column_vector x) const;
            double compute_cost(column_vector& x) const;
            double do_compute_cost(nng::Vector& theta) const;// compute cost function

        private:
            size_t _input_size; // the number of input units
            size_t _hidden_size; // the number of hidden units
            size_t _num_classes;
            se_net_config _net_config;
            double _lambda; // weight decay parameter
            nng::Matrix2d _data;
            nng::Vector _labels;

    };

}
#endif
