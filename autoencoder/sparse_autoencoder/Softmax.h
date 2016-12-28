#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "nng_math.h"
#include <utility>      // std::pair

typedef std::pair <double,nng::Vector> double_vector;
namespace nng{
    
    class Softmax
    {
        public:
            Softmax(size_t num_classes, size_t input_size, double lambda, nng::Matrix2d& data, nng::Vector& labels);
            ~Softmax();
            
            nng::Vector initialize();
            double operator() (column_vector x) const;
            double compute_softmax_cost(column_vector& x) const;
           // double_vector softmax_cost(nng::Vector& theta) const;
			double softmax_cost(nng::Vector& theta) const;
            nng::Vector softmax_predict(nng::Vector& opt_theta, nng::Matrix2d& data);
            //nng::Vector softmax_train(size_t input_size, size_t num_classes, double lambda, nng::Matrix2d& data, nng::Vector& labels);
        private:
            size_t num_classes;
            size_t input_size;
            double lambda;
            nng::Matrix2d data;
            nng::Vector labels;
    };

}

#endif