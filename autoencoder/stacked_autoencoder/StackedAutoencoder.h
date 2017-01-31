#ifndef STACKED_AUTOENCODER_H
#define STACKED_AUTOENCODER_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"
#include "Softmax.h"

namespace nng{


			
    class StackedAutoencoder
    {
        public:
            StackedAutoencoder(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels);
            ~StackedAutoencoder();


            double operator() (column_vector x) const;
            double compute_cost(column_vector& x) const;
            double do_compute_cost(nng::Vector& theta) const;// compute cost function
			nng::Vector stacked_autoencoder_predict(nng::Vector& opt_theta, nng::Matrix2d& data, nng::Softmax& sm);

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
