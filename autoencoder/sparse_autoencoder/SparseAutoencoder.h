#ifndef SPARSEAUTOENCODER_H
#define SPARSEAUTOENCODER_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{
    
    class SparseAutoencoder
    {
        public:
            SparseAutoencoder(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data);
            ~SparseAutoencoder();

            nng::Vector initialize();
            void forward_backward(nng::Vector& theta, nng::Matrix2d& data);// data: matrix containing the training data.  dim of data : visible_size*m. So, data(:,i) is the i-th training example
            //void forward_propagation(const nng::Matrix2d& data);
            //void compute_cost_and_backward_propagation(const nng::Vector& theta, const nng::Matrix2d& data);
            nng::Matrix2d sparse_autoencoder(const nng::Vector& theta, size_t hidden_size, size_t visible_size, const nng::Matrix2d& data);

            
            double operator() (column_vector x) const;
            double compute_cost(column_vector& x) const;
            const column_vector compute_grad(const column_vector& x);
            
            double do_compute_cost(nng::Vector& theta) const;// compute cost function
            nng::Vector do_compute_grad(nng::Vector& theta);// back proparagion compute gradient
            
            nng::Vector get_grad(){return grad;}
			nng::Matrix2d getW1(nng::Vector& theta);
            
			size_t getVisibleSize(){return visible_size;}
			size_t getHiddenSize(){return hidden_size;}
			double getLambda(){return lambda;}
			double getBeta(){return beta;}
			double getRho(){return sparsity_param;}

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
