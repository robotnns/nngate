#include "nng_math.h"
#include "SparseAutoencoder.h"
#include "SparseAutoencoderGrad.h"
#include "Softmax.h"
#include "SoftmaxGrad.h"
#include <iostream>
#include <pixdb.h>
#include "dlib/optimization/optimization.h"


int main(int argc, char **argv) {
	/*size_t patch_size = 1;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 1;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 1;
	//size_t num_labels = 2;

	nng::Matrix2d data(input_size,m,0);

    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta(4,0.0);
	theta(0) = 0.90353282;
	theta(1) = 1.34969621;
	double cost = ae.do_compute_cost(theta);
	cout<<"cost ="<<cost <<endl;
    //theta.print();

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
	
	*/

	/*
	size_t patch_size = 2;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 2;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 1;
	//size_t num_labels = 2;

	nng::Matrix2d data(input_size,m,0);

    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta(22,0.0);
	theta(0) = 0.31545095;
	theta(1) = -0.47100747;
	theta(2) = 0.78706476; 
	theta(3) = -0.18098121;  
	theta(4) = 0.20255144; 
	theta(5) = -0.305009;
	theta(6) = -0.27080672;  
	theta(7) = 0.08962147;  
	theta(8) = 0.74871292;  
	theta(9) = 0.90504337; 
	theta(10) = -0.47377659; 
	theta(11) = -0.73820624;
	theta(12) = -0.78370312;  
	theta(13) = 0.4062228;  
	theta(14) = -0.36713266;  
	theta(15) = 0.57102779; 
	
	double cost = ae.do_compute_cost(theta);
	cout<<"cost ="<<cost <<endl;
    //theta.print();

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
	*/
	
  /*
	size_t patch_size = 2;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 2;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 2;
	//size_t num_labels = 2;

	nng::Matrix2d data(input_size,m,0);

    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta(22,0.0);
	theta(0) = 0.12021754;
	theta(1) = -0.39669575;
	theta(2) = 0.89293814; 
	theta(3) = 0.76081258;  
	theta(4) = -0.31729271; 
	theta(5) = -0.12979209;
	theta(6) = -0.18159828;  
	theta(7) = 0.24300289;  
	theta(8) = -0.35757678;  
	theta(9) = -0.30943324; 
	theta(10) = 0.40818056; 
	theta(11) = -0.92513701;
	theta(12) = 0.11168352;  
	theta(13) = -0.63819158;  
	theta(14) = -0.55413059;  
	theta(15) = -0.79922977; 
	
	double cost = ae.do_compute_cost(theta);
	cout<<"cost ="<<cost <<endl;
    //theta.print();

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
	*/
	
	
	/*
	//test softmax
	size_t patch_size = 2;//28
    size_t input_size = patch_size*patch_size;
    //size_t hidden_size = 2;//196;
    double lambda = 0.003;

	size_t m_train = 3;
	size_t num_labels = 2;

	nng::Matrix2d train_features(input_size,m_train,0);
	nng::Vector train_labels(m_train,0.0);
	train_labels(1) = 1;
	train_labels(2) = 1;


    nng::Vector theta(8,0.0);
	        
         
   
	theta(0) = 0.00743285;
	theta(1) = -0.00119746;
	theta(2) = -0.00736693; 
	theta(3) = -0.00463688;  
	theta(4) = -0.00257648; 
	theta(5) = 0.00178111;
	theta(6) = 0.00077875;  
	theta(7) = -0.00262821;  
	
	nng::Softmax sm(num_labels, input_size, lambda, train_features, train_labels);
	nng::SoftmaxGrad sm_grad(num_labels, input_size, lambda, train_features, train_labels);
	double cost = sm.softmax_cost(theta);
	cout<<"cost ="<<cost <<endl;
    //theta.print();

	nng::Vector grad = sm_grad.softmax_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
	*/
	
	
	
	
	
	
	
	
	
	
	
	
	std::vector <STRU_PIXDB_REC_DOUBLE> v_image_in;	
	pixdb pdb;
	pdb.set_file_name(argv[1]);	
	pdb.read_all(v_image_in);
	//nng::print_v(v_image_in[0].pix_buf);
	
	std::vector <STRU_PIXDB_REC_DOUBLE> v_image_in2;	
	pixdb pdb2;
	pdb2.set_file_name(argv[2]);	
	pdb2.read_all(v_image_in2);

	
	size_t nb_image = v_image_in.size();
	size_t nb_image2 = v_image_in2.size();
	
	size_t patch_size = 28;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 196;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;
    size_t m1 = nb_image / 2;
	size_t m2 = nb_image2 / 2;
	size_t m = m1 + m2;
	std::cout<<"sample number = "<<m<<std::endl;
	

	nng::Matrix2d data(input_size,m,1);
	size_t i_start_x, i_start_y;
	nng::Matrix2d* m_image;
	nng::Matrix2d m_patch(patch_size, patch_size,0);
	for (size_t i = 0; i < m1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		data.set_col(m_patch.toVector(), i);
	}
	for (size_t i = 0; i < m2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		data.set_col(m_patch.toVector(), m1 + i);
	}
	//std::cout<<"data="<<std::endl;
	//data.print();
    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
	std::cout<<"input_size = "<<input_size<<std::endl;
	nng::SparseAutoencoderGrad ae_grad(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta = ae.initialize();
	//cout<<"initial theta ="<<endl;
    //theta.print();

    nng::column_vector x_theta = nng::cnn_vector_to_column_vector(theta);

	double opt_cost;
	cout<<"computing opt cost ... "<<endl;
//    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(1e-1), ae, 
	//						  dlib::derivative([&](nng::column_vector& x){return ae.compute_grad(x);}), x_theta, -1);

    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(1e-10), ae, //dlib::objective_delta_stop_strategy(1e-7).be_verbose()
							  ae_grad, x_theta, -1);//375.9
//    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
//							  dlib::derivative(ae), x_theta, 0.528);
							  
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
    
	nng::Vector opt_theta = nng::column_vector_to_cnn_vector(x_theta);
	//cout<<"opt theta ="<<endl;
	//opt_theta.print();
	//nng::Matrix2d W1 = ae.getW1(opt_theta);
	//W1.print();
	
	
	
	
	size_t m_train_1 = nb_image /4;
	size_t m_train_2 = nb_image /4;
	size_t m_train = m_train_1 + m_train_2;
    nng::Matrix2d train_data(input_size,m_train,1);
	nng::Vector train_labels(m_train,0.0);
	for (size_t i = 0; i < m_train_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_data.set_col(m_patch.toVector(), i);
		train_labels(i) = 0;//v_image_in[i].label;
	}
	
	for (size_t i = 0; i < m_train_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_data.set_col(m_patch.toVector(), i + m_train_1);
		train_labels(i + m_train_1) = 1;//v_image_in2[i].label;
	}	


	cout<<"prepare train data"<<endl;
	nng::Matrix2d train_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, train_data);
	
	size_t m_test_1 = nb_image /20;
	size_t m_test_2 = nb_image /20;
	size_t m_test = m_test_1 + m_test_2;
	
    nng::Matrix2d test_data(input_size,m_test,1);
	for (size_t i = 0; i < m_test_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i + m_train_1].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i);
	}
	
	for (size_t i = 0; i < m_test_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
		i_start_y = (size_t)(nng::rand_a_b(0.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size)));
        m_image = new nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i + m_train_2].pix_buf);
		m_patch = m_image->getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i + m_test_1);
	}	

	cout<<"compute test eatures"<<endl;
	nng::Matrix2d test_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, test_data);
	//test_features.print();
	
	// softmax train
	
	size_t num_labels = 2;
	cout<<"softmax train"<<endl;
	nng::Softmax sm(num_labels, hidden_size, lambda, train_features, train_labels);
	nng::SoftmaxGrad sm_grad(num_labels, hidden_size, lambda, train_features, train_labels);
	cout<<"softmax train initialize"<<endl;
    nng::Vector theta_sm = sm.initialize();
	
	nng::column_vector x_theta_sm = nng::cnn_vector_to_column_vector(theta_sm);
	cout<<"softmax train find min cost"<<endl;
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(1e-20), sm, 
							  sm_grad, x_theta_sm, -1);	
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
	
	nng::Vector opt_theta_sm = nng::column_vector_to_cnn_vector(x_theta_sm);
	nng::Vector predictions = sm.softmax_predict(opt_theta_sm, test_features);
	std::cout<<"predictions ="<<std::endl;
	predictions.print();
  	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
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
