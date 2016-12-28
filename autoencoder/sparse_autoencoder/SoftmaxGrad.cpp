#include "SoftmaxGrad.h"
#include "dlib/optimization/optimization.h"


nng::SoftmaxGrad::SoftmaxGrad(size_t num_classes, size_t input_size, double lambda, nng::Matrix2d& data,  nng::Vector& labels):
	num_classes(num_classes)
	,input_size(input_size)
	,lambda(lambda)
	,data(data)
	,labels(labels)
{
}

nng::SoftmaxGrad::~SoftmaxGrad()
{
}


const nng::column_vector nng::SoftmaxGrad::operator() (nng::column_vector x) const {  return compute_softmax_grad(x); } 

const nng::column_vector nng::SoftmaxGrad::compute_softmax_grad(nng::column_vector& x) const
{
    nng::Vector v = nng::column_vector_to_cnn_vector(x);
	nng::Vector grad = softmax_grad(v);
	nng::column_vector result = nng::cnn_vector_to_column_vector(grad);
    return result;
}

/*
param theta:
param num_classes: the number of classes
param input_size: the size N of input vector (hidden size for autoencoder)
param lambda_: weight decay parameter
param data: the N x M input matrix, where each column corresponds a single test set
param labels: an M x 1 matrix containing the labels for the input data
*/
const nng::Vector nng::SoftmaxGrad::softmax_grad(nng::Vector& theta) const
{
    size_t m = data.get_cols();// data is output of hidden layer, dim data: input_size*m
    nng::Matrix2d m_theta(num_classes, input_size, theta); // y = theta*x, dim theta: num_classes*input_size
    nng::Matrix2d theta_data = m_theta * data; // dim theta_data: num_classes*m
    theta_data = theta_data - theta_data.max(); 
	/*
	nng::Matrix2d theta_data_exp = theta_data.getBlock(0,0,num_classes-1,m).exp();
	nng::Vector theta_data_exp_sum = theta_data_exp.sum(0);
	nng::Vector theta_data_exp_sum_b = 1 + theta_data_exp_sum;
    nng::Matrix2d prob_data_k_1 = theta_data_exp / theta_data_exp_sum_b;// dim prob_data_k_1: (num_classes - 1)*m
	nng::Matrix2d prob_data_k(1,m,1.0);
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < num_classes-1; j++)
			prob_data_k(0,i) -= prob_data_k_1(j,i);
	}
	nng::Matrix2d prob_data = prob_data_k_1.concatenate(prob_data_k,1);*/
	nng::Matrix2d theta_data_exp = theta_data.exp();
	nng::Vector theta_data_exp_sum = theta_data_exp.sum(0);

    nng::Matrix2d prob_data = theta_data_exp / theta_data_exp_sum;
//    nng::Matrix2d prob_data = theta_data_exp / theta_data_exp.sum(0);
    //indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
	//indicator = np.array(indicator.todense())
	nng::Matrix2d indicator(num_classes, m, 0.0); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(labels(i), i) = 1; // the class of the i-th example is labels(i)
	}


    //grad = (-1. / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta
	nng::Matrix2d diff_indicator_prob = indicator - prob_data;
    nng::Matrix2d grad = (-1. / m) * diff_indicator_prob * (data.transpose()) + lambda * m_theta; // dim grad: num_classes*input_size
    //return cost, grad.flatten()		
   
    return grad.to_cnnvector();

}

