#ifndef CNN_H
#define CNN_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{
    
    class Cnn
    {
        public:
            Cnn(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data);
            ~Cnn();

 
        private:



    };

}
#endif
