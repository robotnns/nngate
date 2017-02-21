#include "nng_math.h"
#include "nng_pca.h"
#include "SparseAutoencoder.h"
#include "SparseAutoencoderGrad.h"
#include "Softmax.h"
#include "SoftmaxGrad.h"
#include <iostream>
#include <pixdb.h>
#include "dlib/optimization/optimization.h"
#include <fstream>

bool USE_WHITENING = true;

std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

nng::Vector read_opt_theta_from_file(const char* filename);
nng::Vector autoencoder_compute_theta(nng::SparseAutoencoder& ae, nng::SparseAutoencoderGrad& ae_grad, size_t patch_size);
void softmax_train(nng::Softmax& sm, nng::SoftmaxGrad& sm_grad, nng::Matrix2d& test_features, nng::Vector& test_labels);

int main(int argc, char **argv) 
{
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
	std::cout<<"nb_image = "<<nb_image<<std::endl;
    std::cout<<"nb_image2 = "<<nb_image2<<std::endl;
    
	size_t patch_size = 28;//28
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 196;//196;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;
    size_t m1 = nb_image;//nb_image / 2;
	size_t m2 = nb_image2;//nb_image2 / 2;
	size_t m = m1 + m2;
	std::cout<<"sample number = "<<m<<std::endl;
	

	nng::Matrix2d data(input_size,m);
	size_t i_start_x, i_start_y;
	nng::Matrix2d m_image(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT);
	nng::Matrix2d m_patch(patch_size, patch_size,0);
	for (size_t i = 0; i < m1; i++)
	{

		i_start_x = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
		i_start_y = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		data.set_col(m_patch.toVector(), i);
	}
	for (size_t i = 0; i < m2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
		i_start_y = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
		m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		data.set_col(m_patch.toVector(), m1 + i);
	}
    if (USE_WHITENING)
    {
        std::cout<<"USE_WHITENING"<<std::endl;
        nng::PCA_ZCA zca(data,0.99,0.1);
        data = zca.getZcaWhite();
    }
	//std::cout<<"data="<<std::endl;
	//data.print();
	
    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
	std::cout<<"input_size = "<<input_size<<std::endl;
	nng::SparseAutoencoderGrad ae_grad(input_size, hidden_size, rho, lambda, beta, m,data);
	
	nng::Vector opt_theta = autoencoder_compute_theta(ae,ae_grad, patch_size);
	
	//nng::Vector opt_theta = read_opt_theta_from_file("opt_theta.dat");
	
	size_t m_train_1 = nb_image;//nb_image /4;
	size_t m_train_2 = nb_image;//nb_image /4;
	size_t m_train = m_train_1 + m_train_2;
    nng::Matrix2d train_data(input_size,m_train,1);
	nng::Vector train_labels(m_train,0.0);
	for (size_t i = 0; i < m_train_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
		i_start_y = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_data.set_col(m_patch.toVector(), i);
		train_labels(i) = 0;//v_image_in[i].label;
	}
	
	for (size_t i = 0; i < m_train_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
		i_start_y = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_data.set_col(m_patch.toVector(), i + m_train_1);
		train_labels(i + m_train_1) = 1;//v_image_in2[i].label;
	}	

    if (USE_WHITENING)
    {
        nng::PCA_ZCA zca_train(train_data,0.99,0.1);
        train_data = zca_train.getZcaWhite();
    }
	cout<<"prepare train data"<<endl;
	nng::Matrix2d train_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, train_data);
	
	size_t m_test_1 = nb_image;//nb_image /10;
	size_t m_test_2 = nb_image;//nb_image /10;
	size_t m_test = m_test_1 + m_test_2;
	
    nng::Matrix2d test_data(input_size,m_test,1);
	nng::Vector test_labels(m_test,0.0);
	for (size_t i = 0; i < m_test_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(50.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-50.0)));
		i_start_y = (size_t)(nng::rand_a_b(50.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-50.0)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i + m_train_1].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i);
		test_labels(i) = 0;
	}
	
	for (size_t i = 0; i < m_test_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(50.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-50.0)));
		i_start_y = (size_t)(nng::rand_a_b(50.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-50.0)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i + m_train_2].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i + m_test_1);
		test_labels(i + m_test_1) = 1;
	}	

	cout<<"compute test eatures"<<endl;
	nng::Matrix2d test_features = ae.sparse_autoencoder(opt_theta, hidden_size, input_size, test_data);

	
	// softmax train
	
	size_t num_labels = 2;
	cout<<"softmax train"<<endl;
	double lambda_softmax = 1e-5;
	nng::Softmax sm(num_labels, hidden_size, lambda_softmax, train_features, train_labels);
	nng::SoftmaxGrad sm_grad(num_labels, hidden_size, lambda_softmax, train_features, train_labels);

	softmax_train(sm, sm_grad, test_features, test_labels);
	
	return 0;
}	

nng::Vector read_opt_theta_from_file(const char* filename)
{
  std::cout << "Opening opt_theta file"<<std::endl;
  std::string line;
  std::ifstream opt_theta_file (filename);
  size_t hidden_size = 0;
  size_t input_size = 0;
  size_t total_size = 0;
  
  if (opt_theta_file.is_open())
  {
	std::getline (opt_theta_file,line);
	std::getline (opt_theta_file,line);
	hidden_size = std::stoi(line);
	std::cout << "hidden_size = "<< hidden_size << std::endl;
	std::getline (opt_theta_file,line);
	std::getline (opt_theta_file,line);
	input_size = std::stoi(line);
	std::cout << "input_size = "<< input_size << std::endl;
	std::getline (opt_theta_file,line);
	std::getline (opt_theta_file,line);
	total_size = std::stoi(line);
	std::cout << "total_size = "<< total_size << std::endl;
	
	std::getline (opt_theta_file,line);  
	std::vector<std::string>   opt_theta_str = split(line, ' ');
	assert(total_size == opt_theta_str.size());
	
	nng::Vector opt_theta(total_size,0);
	for (size_t i = 0; i < opt_theta_str.size(); i++)
	{
		opt_theta(i) = std::stod(opt_theta_str[i]);
	}
	
    opt_theta_file.close();
	return opt_theta;
  }

  else
  { 
	  std::cout << "Unable to open opt_theta file"<<std::endl; 	
	  nng::Vector opt_theta(total_size,0);
	  return opt_theta;
  }
  
  
}

nng::Vector autoencoder_compute_theta(nng::SparseAutoencoder& ae, nng::SparseAutoencoderGrad& ae_grad, size_t patch_size)
{

	size_t input_size = ae.getVisibleSize();
	size_t hidden_size = ae.getHiddenSize();
    nng::Vector theta = ae.initialize();
	//cout<<"initial theta ="<<endl;
    //theta.print();

    nng::column_vector x_theta = nng::cnn_vector_to_column_vector(theta);

	double opt_cost;
	cout<<"computing opt cost ... "<<endl;
    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(30), dlib::objective_delta_stop_strategy(1e-10), ae, //dlib::objective_delta_stop_strategy(1e-7).be_verbose()
							  ae_grad, x_theta, -1);//363
//    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
//							  dlib::derivative(ae), x_theta, 0.528);
							  
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
    
	nng::Vector opt_theta = nng::column_vector_to_cnn_vector(x_theta);
	//cout<<"opt theta ="<<endl;
	//opt_theta.print();
	nng::Matrix2d W1 = ae.getW1(opt_theta);
	std::ofstream opt_theta_file;
	opt_theta_file.open ("opt_theta_file.txt");
	opt_theta_file << "hidden_size\n";
	opt_theta_file << hidden_size << "\n";
	opt_theta_file << "input_size\n";
	opt_theta_file << input_size << "\n";
	opt_theta_file << "total_size\n";
	opt_theta_file << opt_theta.get_length() << "\n";
	for (size_t i = 0; i < opt_theta.get_length(); i++)
		opt_theta_file << opt_theta(i) <<" ";
	opt_theta_file << "\n";	
	opt_theta_file.close();
	
	std::ofstream w1_file;
	w1_file.open ("w1_file.txt");
	w1_file << "hidden_size\n";
	w1_file << hidden_size << "\n";
	w1_file << "input_size\n";
	w1_file << input_size << "\n";
	w1_file << "patch_size\n";
	w1_file << patch_size << "\n";
	for (size_t i = 0; i < W1.get_rows(); i++)
		for (size_t j = 0; j < W1.get_cols(); j++)
			w1_file << W1(i,j) <<" ";
	w1_file << "\n";	
	w1_file.close();

	return opt_theta;	
}
	
void softmax_train(nng::Softmax& sm, nng::SoftmaxGrad& sm_grad, nng::Matrix2d& test_features, nng::Vector& test_labels)
{

	cout<<"softmax train initialize"<<endl;
    nng::Vector theta_sm = sm.initialize();
	
	nng::column_vector x_theta_sm = nng::cnn_vector_to_column_vector(theta_sm);
	cout<<"softmax train find min cost"<<endl;
    dlib::find_min(dlib::lbfgs_search_strategy(30), dlib::objective_delta_stop_strategy(1e-30), sm, 
							  sm_grad, x_theta_sm, -1);	
	
	nng::Vector opt_theta_sm = nng::column_vector_to_cnn_vector(x_theta_sm);
	nng::Vector predictions = sm.softmax_predict(opt_theta_sm, test_features);
	std::cout<<"predictions ="<<std::endl;
	predictions.print();
  	
	size_t error_nb = 0;
	size_t m_test = test_labels.get_length();
	for (size_t i = 0; i < m_test; i++)
	{
		if ((size_t)test_labels(i) != (size_t)predictions(i))
			error_nb += 1;
	}

	double accuracy = (double)(m_test - error_nb)/(double)m_test;
	std::cout<<"accuracy = "<<accuracy*100<<"%"<<std::endl;	
}

void test_autoencoder_1()
{
	size_t patch_size = 1;
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 1;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 1;

	nng::Matrix2d data(input_size,m,0);

    nng::SparseAutoencoder ae(input_size, hidden_size, rho, lambda, beta, m,data);
    nng::Vector theta(4,0.0);
	theta(0) = 0.90353282;
	theta(1) = 1.34969621;
	double cost = ae.do_compute_cost(theta);
	cout<<"cost ="<<cost <<endl;

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();	
}

void test_autoencoder_2()
{
	
	size_t patch_size = 2;
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 2;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 1;

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

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
	
}

void test_autoencoder_3()
{

	size_t patch_size = 2;
    size_t input_size = patch_size*patch_size;
    size_t hidden_size = 2;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;

	size_t m = 2;

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

	nng::Vector grad = ae.do_compute_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
		
}
void test_softmax()
{
	size_t patch_size = 2;
    size_t input_size = patch_size*patch_size;
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

	nng::Vector grad = sm_grad.softmax_grad(theta);
	cout<<"grad"<<endl;
	grad.print();
		
}
	
void test_autoencoder_softmax()
{	

	size_t input_size = 4;
	size_t hidden_size = 2;
    double rho = 0.1;
    double lambda = 0.003;
    double beta = 3;
	size_t m = 3;
	size_t num_labels = 2;
	
    nng::Matrix2d data(input_size,m,1);
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
	nng::Vector opt_theta = nng::column_vector_to_cnn_vector(x_theta);
	cout<<"opt theta ="<<endl;
	opt_theta.print();

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
	
	nng::Vector opt_theta_sm = nng::column_vector_to_cnn_vector(x_theta_sm);
	nng::Vector predictions = sm.softmax_predict(opt_theta_sm, test_features);
}

