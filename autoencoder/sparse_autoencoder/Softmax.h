#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "cnn_util.h"
#include <utility>      // std::pair

typedef std::pair <double,CnnVector> double_vector;
class Softmax
{
    public:
        Softmax(size_t num_classes, size_t input_size, double lambda, Matrix2d& data, CnnVector& labels);
        ~Softmax();
		
		CnnVector initialize();
		double operator() (column_vector x) const;
		double compute_softmax_cost(column_vector& x) const;
		double_vector softmax_cost(CnnVector& theta) const;
		CnnVector softmax_predict(CnnVector& opt_theta, Matrix2d& data);
		//CnnVector softmax_train(size_t input_size, size_t num_classes, double lambda, Matrix2d& data, CnnVector& labels);
	private:
		size_t num_classes;
		size_t input_size;
		double lambda;
		Matrix2d data;
		CnnVector labels;
};
#endif