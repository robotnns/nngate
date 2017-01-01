#ifndef SOFTMAXGRAD_H
#define SOFTMAXGRAD_H

#include "nng_math.h"
#include <utility>      // std::pair

namespace nng{
    
    class SoftmaxGrad
    {
        public:
            SoftmaxGrad(size_t num_classes, size_t input_size, double lambda, nng::Matrix2d& data, nng::Vector& labels);
            ~SoftmaxGrad();
            

            const column_vector operator() (column_vector x) const;
            const column_vector compute_softmax_grad(column_vector& x) const;
            const nng::Vector softmax_grad(nng::Vector& theta) const;

         private:
            size_t num_classes;
            size_t input_size;
            double lambda;
            nng::Matrix2d data;
            nng::Vector labels;
    };

}

#endif