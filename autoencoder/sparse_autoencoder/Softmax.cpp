#include "Softmax.h"
#include "dlib/optimization/optimization.h"

Softmax::Softmax(size_t num_classes, size_t input_size, double lambda, Matrix2d& data,  CnnVector& labels):
	num_classes(num_classes)
	,input_size(input_size)
	,lambda(lambda)
	,data(data)
	,labels(labels)
{
}

Softmax::~Softmax()
{
}

CnnVector Softmax::initialize()
{
    //theta = 0.005 * np.random.randn(num_classes * input_size)
    CnnVector theta(num_classes * input_size, 0.0);
    for(size_t  i = 0 ; i < num_classes * input_size ; i ++ )
    {
          theta(i) = 0.005 * util::normal_distribution_rand(0.0,1.0);
    }
    return theta;
}

double Softmax::operator() (column_vector x) const {  return compute_softmax_cost(x); } 

double Softmax::compute_softmax_cost(column_vector& x) const
{
    CnnVector v = column_vector_to_cnn_vector(x);
    return softmax_cost(v).first;
}

/*
param theta:
param num_classes: the number of classes
param input_size: the size N of input vector (hidden size for autoencoder)
param lambda_: weight decay parameter
param data: the N x M input matrix, where each column corresponds a single test set
param labels: an M x 1 matrix containing the labels for the input data
*/
double_vector Softmax::softmax_cost(CnnVector& theta) const
{
    size_t m = data.get_cols();// data is output of hidden layer, dim data: input_size*m
    Matrix2d m_theta(num_classes, input_size, theta); // y = theta*x, dim theta: num_classes*input_size
    Matrix2d theta_data = m_theta * data; // dim theta_data: num_classes*m
    theta_data = theta_data - theta_data.max(); 
	Matrix2d theta_data_exp = theta_data.exp();
    Matrix2d prob_data = theta_data_exp / theta_data_exp.sum(0);
    //indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
	//indicator = np.array(indicator.todense())
	Matrix2d indicator(num_classes, m, 0.0); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(labels(i), i) = 1; // the class of the i-th example is labels(i)
	}
	//cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lambda_ / 2) * np.sum(theta * theta)
    double cost =  (-1 / m) *( (indicator.dot(prob_data)).sum() ) + 0.5 * lambda * (theta.dot(theta)).sum();
    //grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta
	Matrix2d diff_indicator_prob = indicator - prob_data;
    Matrix2d grad = (-1 / m) * diff_indicator_prob * (data.transpose()) + lambda * m_theta; // dim grad: num_classes*input_size
    //return cost, grad.flatten()		
    double_vector result = std::make_pair(cost,grad.to_cnnvector());
    return result;

}

/* data - the N x M input matrix, where each column data(:, i) corresponds to a single test set*/
CnnVector Softmax::softmax_predict(CnnVector& opt_theta, Matrix2d& data)
{

    Matrix2d m_opt_theta(num_classes, input_size, opt_theta);// dim m_opt_theta: num_classes * input_size

    Matrix2d prod = m_opt_theta * data; // dim data: input_size * m, dim prod: num_classes * m
    Matrix2d pred = prod.exp() / prod.exp().sum(0); //dim pred: num_classes * m
    CnnVector prediction = pred.argmax(0); // dim prediction: m

	return prediction;
}

/*softmaxTrain Train a softmax model with the given parameters on the given data. 
Returns softmaxOptTheta, a vector containing the trained parameters for the model.

input_size: the size of an input vector x^(i)
num_classes: the number of classes
lambda: weight decay parameter
input_data: an N by M matrix containing the input data, such that 
inputData(:, c) is the cth input
labels: M by 1 matrix containing the class labels for the corresponding inputs. labels(c) is the class label for
          the cth input
options (optional): options
options.maxIter: number of iterations to train for
*/

/*
CnnVector Softmax::softmax_train(size_t input_size, size_t num_classes, double lambda, Matrix2d& data, CnnVector& labels)
{
    //theta = 0.005 * np.random.randn(num_classes * input_size)
    CnnVector theta(num_classes * input_size, 0.0);
    for(size_t  i = 0 ; i < num_classes * input_size ; i ++ )
    {
          theta(i) = 0.005 * util::normal_distribution_rand(0.0,1.0);
    }
	
    size_t len = theta.get_length();
    column_vector x_theta(len);
    for(size_t i = 0; i < len; ++i)
    {
        x_theta(i) = theta(i);
    }	
 double  opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
							  dlib::derivative([&](column_vector& x){return ae.compute_grad(x);}), x_theta, 0.528);
  //  J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

  //  result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

  //  print result
  //  # Return optimum theta, input size & num classes
  //  opt_theta = result.x

//return opt_theta, input_size, num_classes
}
*/
