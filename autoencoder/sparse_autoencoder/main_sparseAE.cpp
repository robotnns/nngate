#include "nng_math.h"
#include "SparseAutoencoder.h"
#include "Softmax.h"
#include <iostream>
#include <pixdb.h>
#include "dlib/optimization/optimization.h"


int main(int argc, char **argv) {
	
	std::vector <STRU_PIXDB_REC_DOUBLE> v_image_in;	
	pixdb pdb;
	pdb.set_file_name(argv[1]);	
	pdb.read_all(v_image_in);

	
	std::vector <STRU_PIXDB_REC_DOUBLE> v_image_in2;	
	pixdb pdb2;
	pdb2.set_file_name(argv[2]);	
	pdb2.read_all(v_image_in2);

	
	size_t nb_image = v_image_in.size();
	size_t nb_image2 = v_image_in2.size();
	
	size_t patch_size = 8;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 20;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;
    size_t m1 = nb_image / 2;
	size_t m2 = nb_image2 / 2;
	size_t m = m1 + m2;
	//size_t num_labels = 2;

	nng::Matrix2d data(input_size,m,1);
	size_t i_start;
	nng::Matrix2d* m_image;
	nng::Matrix2d m_patch(patch_size, patch_size,0);
	for (size_t i = 0; i < m1; i++)
	{
		i_start = nng::rand_a_b(0, IMG_WIDTH_HEIGHT - patch_size);
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image->getBlock(i_start, i_start, patch_size, patch_size);
		data.set_col(m_patch.toVector(), i);
	}
	for (size_t i = 0; i < m2; i++)
	{
		i_start = nng::rand_a_b(0, IMG_WIDTH_HEIGHT - patch_size);
		m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image->getBlock(i_start, i_start, patch_size, patch_size);
		data.set_col(m_patch.toVector(), m1 + i);
	}
    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta = ae.initialize();
	//cout<<"initial theta ="<<endl;
    //theta.print();
    
    nng::column_vector x_theta = nng::cnn_vector_to_column_vector(theta);

	double opt_cost;
	cout<<"computing opt cost ... "<<endl;
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
							  dlib::derivative([&](nng::column_vector& x){return ae.compute_grad(x);}), x_theta, 0.528);
//    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
//							  dlib::derivative(ae), x_theta, 0.528);
							  
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
    
	nng::Vector opt_theta = nng::column_vector_to_cnn_vector(x_theta);
	//cout<<"opt theta ="<<endl;
	//opt_theta.print();
	nng::Matrix2d W1 = ae.getW1(opt_theta);
	W1.print();
    /*nng::Matrix2d data(input_size,m,1);
	data(1,0) = 2; data(1,1) = 2; 
	data(3,1) = 2; data(3,2) = 2;
    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta = ae.initialize();
	cout<<"initial theta ="<<endl;
    theta.print();
    
    //ae.forward_backward(theta, data);

    
    //nng::Matrix2d a2 = ae.sparse_autoencoder(theta, hidden_size, input_size, data);
    //std::cout<<"a2 = "<<std::endl;
    //a2.print();

    //nng::Vector grad = ae.get_grad();
    //theta = theta + 0.0001*grad;
    //ae.forward_backward(theta, data);
    
    size_t len = theta.get_length();
    nng::column_vector x_theta(len);
    for(size_t i = 0; i < len; ++i)
    {
        x_theta(i) = theta(i);
    }

	double opt_cost;
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
							  dlib::derivative([&](nng::column_vector& x){return ae.compute_grad(x);}), x_theta, 0.528);

	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
	//nng::Vector opt_theta = ae.get_grad();
	//opt_theta.print();
	nng::Vector opt_theta = column_vector_to_cnn_vector(x_theta);
	cout<<"opt theta ="<<endl;
	opt_theta.print();
	*/
    
    /*
	size_t m_train = 2;
    nng::Matrix2d train_data(input_size,m_train,1);
	train_data(1,0) = 2; train_data(1,1) = 2; 
	nng::Vector train_labels(2,0.0);
	train_labels(1) = 1;
	nng::Matrix2d train_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, train_data);
	
	size_t m_test = 2;
    nng::Matrix2d test_data(input_size,m_test,1);
	test_data(3,1) = 2; test_data(3,2) = 2;
	nng::Matrix2d test_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, test_data);
	
	// softmax train
    
	nng::Softmax sm(num_labels, hidden_size, lambda, train_features, train_labels);
	
    nng::Vector theta_sm = sm.initialize();
	
    size_t len_sm = theta_sm.get_length();
    nng::column_vector x_theta_sm(len_sm);
    for(size_t i = 0; i < len_sm; ++i)
    {
        x_theta_sm(i) = theta_sm(i);
    }	
	
	
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), sm, 
							  dlib::derivative(sm), x_theta_sm, 0.528);	
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
	
	nng::Vector opt_theta_sm = column_vector_to_cnn_vector(x_theta_sm);
	nng::Vector predictions = sm.softmax_predict(opt_theta_sm, test_features);
	*/
	return 0;
}
