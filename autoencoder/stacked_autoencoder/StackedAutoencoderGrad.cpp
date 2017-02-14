#include "StackedAutoencoderGrad.h"
#include <math.h>
#include <iostream>


nng::StackedAutoencoderGrad::StackedAutoencoderGrad(size_t input_size, size_t hidden_size, size_t num_classes, nng::se_net_config net_config, double lambda, nng::Matrix2d& data, nng::Vector& labels):
    _input_size(input_size)
    ,_hidden_size(hidden_size)
    ,_num_classes(num_classes)
    ,_net_config(net_config)
    ,_lambda(lambda)
    ,_data(data)
    ,_labels(labels)
{
}

nng::StackedAutoencoderGrad::~StackedAutoencoderGrad()
{
}

const nng::column_vector nng::StackedAutoencoderGrad::operator() (const nng::column_vector x) const 
{  
	return compute_grad(x); 
} 



const nng::column_vector nng::StackedAutoencoderGrad::compute_grad(const nng::column_vector& x) const
{
    nng::Vector v = nng::column_vector_to_cnn_vector(x);
    nng::Vector vgrad = do_compute_grad(v);
    nng::column_vector result = nng::cnn_vector_to_column_vector(vgrad);
    return result;
}
        
//dim theta: num_classes*hidden_size_L2 + hidden_size_L1*input_size + 1 + hidden_size_L2*hidden_size_L1 + 1
const nng::Vector nng::StackedAutoencoderGrad::do_compute_grad(nng::Vector& theta) const
{

    //std::cout << "StackedAutoencoderGrad: extract the part which compute the softmax gradient" << std::endl;
    nng::Matrix2d softmax_theta(_num_classes, _hidden_size, theta.getSegment(0,_hidden_size*_num_classes));//dim softmax_theta: num_classes*hidden_size_L2

    //std::cout << "StackedAutoencoderGrad: Extract out the stack" << std::endl;
	nng::Vector params = theta.getSegment(_hidden_size*_num_classes,theta.get_length() - _hidden_size*_num_classes);//dim params:hidden_size_L1*input_size + 1 + hidden_size_L2*hidden_size_L1 + 1
	nng::se_net_config net_config = _net_config;//input_size, layer_sizes
	nng::param_config param_net_config(params,net_config);
    nng::se_stack stack = nng::params2stack(param_net_config);//(w1_ae1,b1_ae1),(w1_ae2,b1_ae2), 
																// dim stack[0].first:hidden_size_L1*input_size
																// dim stack[i].first:hidden_size_L2*hidden_size_L1

    size_t m = _data.get_cols();

    //std::cout << "StackedAutoencoderGrad: Forward propagation" << std::endl;
    std::vector<nng::Matrix2d> a;
	a.push_back(_data);//dim data = dim a[0]:input_size,m
    std::vector<nng::Matrix2d> z; 
	z.push_back(nng::Matrix2d(_data.get_rows(),m)); // Dummy value

    for (std::pair<nng::Matrix2d, double> s : stack)
	{
		//std::cout << s.first.get_rows()<< " " << s.first.get_cols()<< std::endl; // 196 784, 196 196
		//std::cout << a.back().get_rows()<< " " << a.back().get_cols()<< std::endl;// 784 612, 196 612
        z.push_back(s.first*(a.back()) + s.second); // dim z[1]:hidden_size_L1*m, dim z[2]: hidden_size_L2*m //196 612, 196 612
        a.push_back(z.back().sigmoid()); // dim a[1]:hidden_size_L1*m, dim a[2]:hidden_size_L2*m
	}
    //std::cout << "StackedAutoencoderGrad: Softmax" << std::endl;
    nng::Matrix2d prod = softmax_theta*(a.back()); // dim prod: num_classes*m
    prod = prod - prod.max();
	
    nng::Matrix2d prob = prod.exp() / prod.exp().sum(0);
	nng::Matrix2d indicator(_num_classes, m); // dim indicator: num_classes*m
	for (size_t i = 0; i < m; ++i)
	{
		indicator(_labels(i), i) = 1.0; // the class of the i-th example is labels(i)
	}
	
	nng::Matrix2d diff_indicator_prob = indicator - prob;
    nng::Matrix2d softmax_grad = (-1.0 / m) * diff_indicator_prob * (a.back().transpose()) + _lambda * softmax_theta;//dim softmax_grad:num_classes*hidden_size_L2
	

    //std::cout << "StackedAutoencoderGrad: Backprop" << std::endl;
    //std::cout << "StackedAutoencoderGrad: Compute partial of cost (J) w.r.t to outputs of last layer (before softmax)" << std::endl;
    nng::Matrix2d softmax_grad_a = softmax_theta.transpose()*diff_indicator_prob; // dim softmax_grad_a: hidden_size_L2 * m

	//std::cout << "StackedAutoencoderGrad: Compute deltas" << std::endl;
	std::vector<nng::Matrix2d> delta;
	delta.push_back(-softmax_grad_a.dot(z.back().sigmoid_prime())); // dim delta[0]: hidden_size_L2 * m
    //delta = [-softmax_grad_a * sigmoid_prime(z[-1])]
	//std::cout << "StackedAutoencoderGrad: Compute d" << std::endl;
	nng::Matrix2d d(1,1);
	std::vector<nng::Matrix2d>::iterator it;
	for (int i = stack.size()-1; i>=0; i--)
	{
		//std::cout << stack[i].first.get_rows()<< " " << stack[i].first.get_cols()<< std::endl;// 196 196, 196 784
		//std::cout << delta[0].get_rows()<< " " << delta[0].get_cols()<< std::endl;//196 612, 196 612
		d = stack[i].first.transpose()*delta[0]; //i=1: hidden_size_L1*m, i=0:input_size*m
		//std::cout << d.get_rows()<< " " << d.get_cols()<< std::endl;//196 612, 784 612
		//std::cout << z[i].get_rows() << " " << z[i].get_cols() << std::endl;//196 612, 196 612

		d = d.dot(z[i].sigmoid_prime());

		it = delta.begin();
		delta.insert(it,d); //i=1,dim delta[0]:hidden_size_L1*m, 
							//i=0, dim delta[0]:input_size*m, dim delta[1]:hidden_size_L1*m, dim delta[2]:hidden_size_L2 * m
	}
    //for i in reversed(range(len(stack))):
      //  d = stack[i]['w'].transpose().dot(delta[0]) * sigmoid_prime(z[i])
      //  delta.insert(0, d)

    //std::cout << "StackedAutoencoderGrad: Compute gradients" << std::endl;
	nng::se_stack stack_grad;
	nng::Matrix2d grad_w_i(1,1);
	nng::Vector grad_b_i(1);
	for (size_t i = 0; i<stack.size(); i++)
	{
		grad_w_i = (delta[i+1]*(a[i].transpose()))/(double)m;// dim grad_w_0: hidden_size_L1*input_size,
															// dim grad_w_1: hidden_size_L2*hidden_size_L1
		grad_b_i = delta[i+1].sum(1)/(double)m; // dim grad_b_0:hidden_size_L1, dim grad_b_1:hidden_size_L2
		
		stack_grad.push_back(std::pair<nng::Matrix2d, double>(grad_w_i,grad_b_i[0]));
	}

	nng::param_config param_net_config_grad = nng::stack2params(stack_grad);
	nng::Vector grad_params = param_net_config_grad.params;
	nng::Vector grad = softmax_grad.to_cnnvector().concatenate(grad_params);
	//dim grad:num_classes*hidden_size_L2 + hidden_size_L1*input_size + hidden_size_L1 + hidden_size_L2*hidden_size_L1 + hidden_size_L2
 
    return grad;
}