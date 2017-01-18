#ifndef STACKEDAUTOENCODERGRAD_H
#define STACKEDAUTOENCODERGRAD_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"
#include "StackedAutoencoder.h"


namespace nng{
			
    class StackedAutoencoderGrad
    {
        public:
            StackedAutoencoderGrad(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels);
            ~StackedAutoencoderGrad();
			
            const column_vector operator() (const column_vector x) const;
            const column_vector compute_grad(const column_vector& x) const;
            const nng::Vector do_compute_grad(nng::Vector& theta) const;// back proparagion compute gradient

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