#include "Softmax.h"
#include "dlib/optimization/optimization.h"


nng::Softmax::Softmax(size_t num_classes, size_t input_size, double lambda, nng::Matrix2d& data,  nng::Vector& labels):
	num_classes(num_classes)
	,input_size(input_size)
	,lambda(lambda)
	,data(data)
	,labels(labels)
{
}

nng::Softmax::~Softmax()
{
}

nng::Vector nng::Softmax::initialize()
{
    //theta = 0.005 * np.random.randn(num_classes * input_size)
    nng::Vector theta(num_classes * input_size, 0.0);
    for(size_t  i = 0 ; i < num_classes * input_size ; i ++ )
    {
          theta(i) = 0.005 * nng::normal_distribution_rand(0.0,1.0);
    }
    return theta;
}

double nng::Softmax::operator() (nng::column_vector x) const {  return compute_softmax_cost(x); } 

double nng::Softmax::compute_softmax_cost(nng::column_vector& x) const
{
    nng::Vector v = nng::column_vector_to_cnn_vector(x);
	//double_vector result = softmax_cost(v);
	//double cost = result.first;
	double cost = softmax_cost(v);
	std::cout<<cost<<std::endl;
    return cost;
}

/*
param theta:
param num_classes: the number of classes
param input_size: the size N of input vector (hidden size for autoencoder)
param lambda_: weight decay parameter
param data: the N x M input matrix, where each column corresponds a single test set
param labels: an M x 1 matrix containing the labels for the input data
*/
double nng::Softmax::softmax_cost(nng::Vector& theta) const
{
    size_t m = data.get_cols();// data is output of hidden layer, dim data: input_size*m
    nng::Matrix2d m_theta(num_classes, input_size, theta); // y = theta*x, dim theta: num_classes*input_size
    nng::Matrix2d theta_data = m_theta * data; // dim theta_data: num_classes*m
    theta_data = theta_data - theta_data.max(); 

/*
	nng::Matrix2d theta_data_exp = theta_data.getBlock(0,0,num_classes-1,m).exp();
	nng::Vector theta_data_exp_sum = theta_data_exp.sum(0);
	nng::Vector theta_data_exp_sum_b = 1.0 + theta_data_exp_sum;
    nng::Matrix2d prob_data_k_1 = theta_data_exp / theta_data_exp_sum_b;// dim prob_data_k_1: (num_classes - 1)*m
	nng::Matrix2d prob_data_k(1,m,1.0);
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < num_classes-1; j++)
			prob_data_k(0,i) -= prob_data_k_1(j,i);
	}
	nng::Matrix2d prob_data = prob_data_k_1.concatenate(prob_data_k,1);
	*/
	
	nng::Matrix2d theta_data_exp = theta_data.exp();
	nng::Vector theta_data_exp_sum = theta_data_exp.sum(0);
    nng::Matrix2d prob_data = theta_data_exp / theta_data_exp_sum;

	nng::Matrix2d indicator(num_classes, m, 0.0); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(labels(i), i) = 1.0; // the class of the i-th example is labels(i)
	}


    double cost =  (-1.0 / m) *( (indicator.dot(prob_data.log())).sum() ) + 0.5 * lambda * (theta.dot(theta)).sum();

	//nng::Matrix2d diff_indicator_prob = indicator - prob_data;
    //nng::Matrix2d grad = (-1.0 / m) * diff_indicator_prob * (data.transpose()) + lambda * m_theta; // dim grad: num_classes*input_size
    //return cost, grad.flatten()		
    //double_vector result = std::make_pair(cost,grad.to_cnnvector());
    return cost;

}

/* data - the N x M input matrix, where each column data(:, i) corresponds to a single test set*/
nng::Vector nng::Softmax::softmax_predict(nng::Vector& opt_theta, nng::Matrix2d& data)
{

    nng::Matrix2d m_opt_theta(num_classes, input_size, opt_theta);// dim m_opt_theta: num_classes * input_size
    nng::Matrix2d prod = m_opt_theta * data; // dim data: input_size * m, dim prod: num_classes * m
    nng::Matrix2d pred = prod.exp() / prod.exp().sum(0); //dim pred: num_classes * m
/*
prod = prod - prod.max();
	size_t m = data.get_cols();
	nng::Matrix2d theta_data_exp = prod.getBlock(0,0,num_classes-1,m).exp();
	nng::Vector theta_data_exp_sum = theta_data_exp.sum(0);
	nng::Vector theta_data_exp_sum_b = 1.0 + theta_data_exp_sum;
    nng::Matrix2d prob_data_k_1 = theta_data_exp / theta_data_exp_sum_b;// dim prob_data_k_1: (num_classes - 1)*m
	nng::Matrix2d prob_data_k(1,m,1.0);
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < num_classes-1; j++)
			prob_data_k(0,i) -= prob_data_k_1(j,i);
	}
	nng::Matrix2d pred = prob_data_k_1.concatenate(prob_data_k,1);
	*/
	nng::Vector prediction = pred.argmax(0); // dim prediction: m
	return prediction;
}
