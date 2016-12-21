#include "cnn_util.h"
#include "SparseAutoencoder.h"
#include "Softmax.h"
#include <iostream>
#include "dlib/optimization/optimization.h"

using namespace std;
int main(int argc, char **argv) {

    size_t input_size = 4;
    size_t hidden_size = 2;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;
    size_t m = 2;
	size_t num_labels = 2;

    Matrix2d data(input_size,m,1);
	data(1,0) = 2; data(1,1) = 2; 
	data(3,1) = 2; data(3,2) = 2;
    SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    CnnVector theta = ae.initialize();
	cout<<"initial theta ="<<endl;
    theta.print();
    
    //ae.forward_backward(theta, data);

    /*
    Matrix2d a2 = ae.sparse_autoencoder(theta, hidden_size, input_size, data);
    std::cout<<"a2 = "<<std::endl;
    a2.print();

    CnnVector grad = ae.get_grad();
    theta = theta + 0.0001*grad;
    ae.forward_backward(theta, data);
    */
    size_t len = theta.get_length();
    column_vector x_theta(len);
    for(size_t i = 0; i < len; ++i)
    {
        x_theta(i) = theta(i);
    }

	double opt_cost;
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
							  dlib::derivative([&](column_vector& x){return ae.compute_grad(x);}), x_theta, 0.528);

	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
	//CnnVector opt_theta = ae.get_grad();
	//opt_theta.print();
	CnnVector opt_theta = column_vector_to_cnn_vector(x_theta);
	cout<<"opt theta ="<<endl;
	opt_theta.print();
	
	size_t m_train = 2;
    Matrix2d train_data(input_size,m_train,1);
	train_data(1,0) = 2; train_data(1,1) = 2; 
	CnnVector train_labels(2,0.0);
	train_labels(1) = 1;
	Matrix2d train_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, train_data);
	
	size_t m_test = 2;
    Matrix2d test_data(input_size,m_test,1);
	test_data(3,1) = 2; test_data(3,2) = 2;
	Matrix2d test_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, test_data);
	
	// softmax train
    
	Softmax sm(num_labels, hidden_size, lambda, train_features, train_labels);
	
    CnnVector theta_sm = sm.initialize();
	
    size_t len_sm = theta_sm.get_length();
    column_vector x_theta_sm(len_sm);
    for(size_t i = 0; i < len_sm; ++i)
    {
        x_theta_sm(i) = theta_sm(i);
    }	
	
	
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), sm, 
							  dlib::derivative(sm), x_theta_sm, 0.528);	
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
	
	CnnVector opt_theta_sm = column_vector_to_cnn_vector(x_theta_sm);
	CnnVector predictions = sm.softmax_predict(opt_theta_sm, test_features);
	return 0;
}
