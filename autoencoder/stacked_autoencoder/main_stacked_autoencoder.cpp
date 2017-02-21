#include "nng_math.h"

#include "SparseAutoencoder.h"
#include "SparseAutoencoderGrad.h"
#include "Softmax.h"
#include "SoftmaxGrad.h"
#include "StackedAutoencoder.h"
#include "StackedAutoencoderGrad.h"
#include <iostream>
#include <pixdb.h>
#include "dlib/optimization/optimization.h"
#include <fstream>


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
nng::Vector autoencoder_compute_theta(nng::SparseAutoencoder& ae, nng::SparseAutoencoderGrad& ae_grad, size_t patch_size, const std::string& filename, double min = -1);
nng::Vector softmax_train(nng::Softmax& sm, nng::SoftmaxGrad& sm_grad, nng::Matrix2d& test_features, nng::Vector& test_labels);
//nng::Vector stacked_autoencoder_compute_theta(nng::StackedAutoencoder& ae, nng::StackedAutoencoderGrad& ae_grad);

int main(int argc, char **argv) 
{
	/*
	size_t input_size = 1*1;
	size_t hidden_size = 1;
	size_t num_classes = 1;
	nng::se_net_config net_config;
	double lambda = 0.003;
	size_t m = 2;
	nng::Matrix2d data(input_size,m,1);
	nng::Vector labels(m,0.0);
    nng::StackedAutoencoder stacked_ae(input_size, hidden_size, num_classes, net_config, lambda, data, labels);
	*/
	size_t patch_size = 48;
	size_t input_size = patch_size*patch_size;
	size_t num_classes = 2;
	size_t hidden_size_L1 = 196;
	size_t hidden_size_L2 = 196;
	double sparsity_param = 0.1;
	double lambda = 0.003;
	double beta = 3;
	double margin = 40;
	//read train_images data
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
	
    size_t m_train_1 = nb_image/100*99;
	size_t m_train_2 = nb_image2/100*99;
	size_t m = m_train_1 + m_train_2;
	std::cout<<"nb of image 1 = "<<nb_image<<std::endl;
	std::cout<<"nb of image 2 = "<<nb_image2<<std::endl;
	std::cout<<"nb of training example 1 = "<<m_train_1<<std::endl;
	std::cout<<"nb of training example 2 = "<<m_train_2<<std::endl;
	nng::Matrix2d train_images(input_size,m,1);
	nng::Vector train_labels(m,0.0);
	size_t i_start_x, i_start_y;
	nng::Matrix2d* m_image(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT);
	nng::Matrix2d m_patch(patch_size, patch_size,0);
	for (size_t i = 0; i < m_train_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
		i_start_y = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_images.set_col(m_patch.toVector(), i);
		train_labels(i) = 0;
	}
	for (size_t i = 0; i < m_train_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
		i_start_y = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
		m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		train_images.set_col(m_patch.toVector(), m_train_1 + i);
		train_labels(i+ m_train_1) = 1;
	}
	
	
	size_t m_test_1 = nb_image /10;
	size_t m_test_2 = nb_image /10;
	size_t m_test = m_test_1 + m_test_2;
	
    nng::Matrix2d test_data(input_size,m_test,1);
	nng::Vector test_labels(m_test,0.0);
	for (size_t i = 0; i < m_test_1; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
		i_start_y = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[i + m_train_1].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i);
		test_labels(i) = 0;
	}
	
	for (size_t i = 0; i < m_test_2; i++)
	{
		i_start_x = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
		i_start_y = (size_t)(nng::rand_a_b(margin, 1.0*(IMG_WIDTH_HEIGHT - patch_size-margin)));
        m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in2[i + m_train_2].pix_buf);
		m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
		test_data.set_col(m_patch.toVector(), i + m_test_1);
		test_labels(i + m_test_1) = 1;
	}
	
	m_image = 0;
	// train autoencoder 1
	nng::SparseAutoencoder ae1(input_size, hidden_size_L1, sparsity_param, lambda, beta, m,train_images);
	nng::SparseAutoencoderGrad ae1_grad(input_size, hidden_size_L1, sparsity_param, lambda, beta, m,train_images);
	nng::Vector sae1_opt_theta = autoencoder_compute_theta(ae1,ae1_grad, patch_size,"opt_theta_file_ae1.dat", -1);//852//sae1_opt_theta=[w1_ae1,w2_ae1,b1_ae1,b2_ae1]
	//nng::Vector sae1_opt_theta = read_opt_theta_from_file("opt_theta_file_ae1.dat");
	
	// train autoencoder 2
	nng::Matrix2d sae1_features = ae1.sparse_autoencoder(sae1_opt_theta, hidden_size_L1, input_size, train_images);
	nng::SparseAutoencoder ae2(hidden_size_L1, hidden_size_L2, sparsity_param, lambda, beta, m,sae1_features);
	nng::SparseAutoencoderGrad ae2_grad(hidden_size_L1, hidden_size_L2, sparsity_param, lambda, beta, m,sae1_features);	
	nng::Vector sae2_opt_theta = autoencoder_compute_theta(ae2,ae2_grad, 14,"opt_theta_file_ae2.dat",-1);//sae2_opt_theta=[w1_ae2,w2_ae2,b1_ae2,b2_ae2]
	//nng::Vector sae2_opt_theta = read_opt_theta_from_file("opt_theta_file_ae2.dat");
	
	// train softmax classifier
	nng::Matrix2d sae2_features = ae2.sparse_autoencoder(sae2_opt_theta, hidden_size_L2, hidden_size_L1, sae1_features);
	double lambda_softmax = 1e-5;
	nng::Softmax sm(num_classes, hidden_size_L2, lambda_softmax, sae2_features, train_labels);
	nng::SoftmaxGrad sm_grad(num_classes, hidden_size_L2, lambda_softmax, sae2_features, train_labels);

	nng::Vector softmax_theta = softmax_train(sm, sm_grad, sae2_features, train_labels);//dim softmax_theta:num_classes*hidden_size_L2
	//size_t softmax_input_size = hidden_size_L2;
	//size_t softmax_num_classes = num_classes;
	
	// Finetune softmax model
	std::cout << "Initialize the stack using the parameters learned" << std::endl;
	nng::se_stack stack;

	nng::Matrix2d w1_ae1(hidden_size_L1, input_size, sae1_opt_theta.getSegment(0,hidden_size_L1*input_size));//dim w1_ae1:hidden_size_L1*input_size
	double b1_ae1 = sae1_opt_theta(2*hidden_size_L1*input_size);
	nng::Matrix2d w1_ae2(hidden_size_L2, hidden_size_L1, sae2_opt_theta.getSegment(0,hidden_size_L2*hidden_size_L1));//dim w1_ae2:hidden_size_L2*hidden_size_L1
	double b1_ae2 = sae2_opt_theta(2*hidden_size_L2*hidden_size_L1);	
	stack.push_back(std::pair<nng::Matrix2d, double>(w1_ae1, b1_ae1));
	stack.push_back(std::pair<nng::Matrix2d, double>(w1_ae2, b1_ae2));
	std::cout << "Initialize the parameters for the deep model" << std::endl;
	nng::param_config param_net_config = nng::stack2params(stack);
	nng::Vector stack_params = param_net_config.params;//stack_params=[w1_ae1,b1_ae1,w1_ae2,b1_ae2]
													   //dim stack_params: hidden_size_L1*input_size + 1 + hidden_size_L2*hidden_size_L1 + 1
	nng::se_net_config net_config = param_net_config.net_config;//input_size, layer_sizes
	nng::Vector stacked_autoencoder_theta = softmax_theta.concatenate(stack_params);// stacked_autoencoder_theta=[softmax_theta,w1_ae1,b1_ae1,w1_ae2,b1_ae2]
	//dim stacked_autoencoder_theta: num_classes*hidden_size_L2 + hidden_size_L1*input_size + 1 + hidden_size_L2*hidden_size_L1 + 1
	nng::StackedAutoencoder sae(input_size, hidden_size_L2, num_classes, net_config, lambda, train_images, train_labels);
	nng::StackedAutoencoderGrad sae_grad(input_size, hidden_size_L2, num_classes, net_config, lambda, train_images, train_labels);
	//nng::Vector stacked_autoencoder_opt_theta = stacked_autoencoder_compute_theta(sae,sae_grad);
	
	nng::column_vector x_stacked_autoencoder_theta = nng::cnn_vector_to_column_vector(stacked_autoencoder_theta);
/*	sae(x_stacked_autoencoder_theta);
	sae_grad(x_stacked_autoencoder_theta);
	sae(x_stacked_autoencoder_theta);
	sae_grad(x_stacked_autoencoder_theta);*/
	double stacked_autoencoder_opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(1e-7), sae, 
														sae_grad, x_stacked_autoencoder_theta, -1);

	std::cout<<"stacked_autoencoder_opt_cost"<<std::endl;
	std::cout<<stacked_autoencoder_opt_cost<<std::endl;
	nng::Vector stacked_autoencoder_opt_theta = nng::column_vector_to_cnn_vector(x_stacked_autoencoder_theta);
	
	std::ofstream opt_theta_file_sae;
	opt_theta_file_sae.open ("opt_theta_file_sae.dat");
	opt_theta_file_sae << "num_classes";
	opt_theta_file_sae << num_classes << "\n";
	opt_theta_file_sae << "hidden_size_L1\n";
	opt_theta_file_sae << hidden_size_L1 << "\n";
	opt_theta_file_sae << "hidden_size_L2\n";
	opt_theta_file_sae << hidden_size_L2 << "\n";
	opt_theta_file_sae << "input_size\n";
	opt_theta_file_sae << input_size << "\n";
	opt_theta_file_sae << "total_size\n";
	opt_theta_file_sae << stacked_autoencoder_opt_theta.get_length() << "\n";
	for (size_t i = 0; i < stacked_autoencoder_opt_theta.get_length(); i++)
		opt_theta_file_sae << stacked_autoencoder_opt_theta(i) <<" ";
	opt_theta_file_sae << "\n";	
	opt_theta_file_sae.close();
	
	
	nng::Vector predictions = sae.stacked_autoencoder_predict(stacked_autoencoder_opt_theta, test_data, sm);
	std::cout<<"predictions ="<<std::endl;
	predictions.print();
  	
	size_t error_nb = 0;
	//size_t m_test = test_labels.get_length();
	for (size_t i = 0; i < m_test; i++)
	{
		if ((size_t)test_labels(i) != (size_t)predictions(i))
			error_nb += 1;
	}

	double accuracy = (double)(m_test - error_nb)/(double)m_test;
	std::cout<<"accuracy = "<<accuracy*100<<"%"<<std::endl;	
	
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

nng::Vector softmax_train(nng::Softmax& sm, nng::SoftmaxGrad& sm_grad, nng::Matrix2d& test_features, nng::Vector& test_labels)
{

	cout<<"softmax train initialize"<<endl;
    nng::Vector theta_sm = sm.initialize();
	
	nng::column_vector x_theta_sm = nng::cnn_vector_to_column_vector(theta_sm);
	cout<<"softmax train find min cost"<<endl;
    dlib::find_min(dlib::lbfgs_search_strategy(30), dlib::objective_delta_stop_strategy(1e-30), sm, 
							  sm_grad, x_theta_sm, -1);	
	
	nng::Vector opt_theta_sm = nng::column_vector_to_cnn_vector(x_theta_sm);
	return opt_theta_sm;
}

nng::Vector autoencoder_compute_theta(nng::SparseAutoencoder& ae, nng::SparseAutoencoderGrad& ae_grad, size_t patch_size, const std::string& filename, double min)
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
							  ae_grad, x_theta, min);//363
//    opt_cost = dlib::find_min(dlib::lbfgs_search_strategy(10), dlib::objective_delta_stop_strategy(), ae, 
//							  dlib::derivative(ae), x_theta, 0.528);
							  
	cout<<"opt cost ="<<endl;
    std::cout<<opt_cost <<std::endl;
    
	nng::Vector opt_theta = nng::column_vector_to_cnn_vector(x_theta);
	//cout<<"opt theta ="<<endl;
	//opt_theta.print();
	nng::Matrix2d W1 = ae.getW1(opt_theta);
	std::ofstream opt_theta_file;
	opt_theta_file.open (filename);
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
	w1_file.open ("w1_"+filename);
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

//nng::Vector stacked_autoencoder_compute_theta(nng::StackedAutoencoder& ae, nng::StackedAutoencoderGrad& ae_grad)
//{
	
//}