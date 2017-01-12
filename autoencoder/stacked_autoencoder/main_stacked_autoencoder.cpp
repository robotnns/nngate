#include "nng_math.h"

//#include "SparseAutoencoder.h"
//#include "SparseAutoencoderGrad.h"
//#include "Softmax.h"
//#include "SoftmaxGrad.h"
#include <iostream>
#include <pixdb.h>
#include "dlib/optimization/optimization.h"
#include <fstream>
#include "stacked_autoencoder.h"


int main(int argc, char **argv) 
{
	size_t input_size = 1*1;
	size_t hidden_size = 1;
	size_t num_classes = 1;
	nng::se_net_config net_config;
	double lambda = 0.003;
	size_t m = 2;
	nng::Matrix2d data(input_size,m,1);
	nng::Vector labels(m,0.0);
    nng::StackedAutoencoder stacked_ae(input_size, hidden_size, num_classes, net_config, lambda, data, labels);
	
	return 0;
}	
