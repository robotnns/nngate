#ifndef SPARSEAUTOENCODER_H
#define SPARSEAUTOENCODER_H

#include "cnn_util.h"
#include "dlib/matrix/matrix.h"



class SparseAutoencoder
{
    public:
        SparseAutoencoder(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, Matrix2d& data);
        ~SparseAutoencoder();

        CnnVector initialize();
        void forward_backward(CnnVector& theta, Matrix2d& data);// data: matrix containing the training data.  dim of data : visible_size*m. So, data(:,i) is the i-th training example
        //void forward_propagation(const Matrix2d& data);
        //void compute_cost_and_backward_propagation(const CnnVector& theta, const Matrix2d& data);
        Matrix2d sparse_autoencoder(const CnnVector& theta, size_t hidden_size, size_t visible_size, const Matrix2d& data);

        
        double operator() (column_vector x) const;
        double compute_cost(column_vector& x) const;
        const column_vector compute_grad(const column_vector& x);
		
        double do_compute_cost(CnnVector& theta) const;// compute cost function
        CnnVector do_compute_grad(CnnVector& theta);// back proparagion compute gradient
		
        CnnVector get_grad(){return grad;}
		

	private:
		size_t visible_size; // the number of input units
		size_t hidden_size; // the number of hidden units
		double sparsity_param; // sparsity parameter. The desired average activation for the hidden units.
		double lambda; // weight decay parameter
		double beta; // weight of sparsity penalty term
        CnnVector grad;
        Matrix2d data;


};

#endif
