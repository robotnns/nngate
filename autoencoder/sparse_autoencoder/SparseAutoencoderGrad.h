#ifndef SPARSEAUTOENCODERGRAD_H
#define SPARSEAUTOENCODERGRAD_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{
    
    class SparseAutoencoderGrad
    {
        public:
            SparseAutoencoderGrad(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data);
            ~SparseAutoencoderGrad();
		
            const column_vector operator() (const column_vector x) const;

            const column_vector compute_grad(const column_vector& x) const;
            
            const nng::Vector do_compute_grad(nng::Vector& theta) const;// back proparagion compute gradient
            
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
 
            nng::Matrix2d data;


    };

}
#endif
