#ifndef STACKED_AUTOENCODER_H
#define STACKED_AUTOENCODER_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{

	
typedef se_stack std::vector<std::pair<Matrix2d, double>>; // vector of (w_i,b_i)

struct se_net_config
{
	size_t input_size;
	std::vector<size_t> layer_sizes;
}

struct param_config
{
	Vectord params; // vector of w_i b_i
	se_net_config  net_config;
}


    class StackedAutoencoder
    {
        public:
            StackedAutoencoder(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data);
            ~StackedAutoencoder();

            param_config stack2params(se_stack& stack);
			se_stack params2stack(param_config& param_net_config);

        private:
            size_t visible_size; // the number of input units
            size_t hidden_size; // the number of hidden units
            double sparsity_param; // sparsity parameter. The desired average activation for the hidden units.
            double lambda; // weight decay parameter
            double beta; // weight of sparsity penalty term
            nng::Vector grad;
            nng::Matrix2d data;


    };

}
#endif
