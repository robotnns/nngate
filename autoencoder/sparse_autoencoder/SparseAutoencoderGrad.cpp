#include "SparseAutoencoderGrad.h"
#include <math.h>
#include <iostream>


nng::SparseAutoencoderGrad::SparseAutoencoderGrad(size_t visible_size, size_t hidden_size, double sparsity_param, double lambda, double beta, size_t m, nng::Matrix2d& data):
    visible_size(visible_size)
    ,hidden_size(hidden_size)
    ,sparsity_param(sparsity_param)
    ,lambda(lambda)
    ,beta(beta)
    ,data(data)
{
}

nng::SparseAutoencoderGrad::~SparseAutoencoderGrad()
{
}


nng::Matrix2d nng::SparseAutoencoderGrad::getW1(nng::Vector& theta)
{
	nng::Matrix2d W1(hidden_size,visible_size,theta.getSegment(0,hidden_size*visible_size));
	return W1;
}

const nng::column_vector nng::SparseAutoencoderGrad::operator() (const nng::column_vector x) const {  return compute_grad(x); } 



const nng::column_vector nng::SparseAutoencoderGrad::compute_grad(const nng::column_vector& x) const
{

    nng::Vector v = nng::column_vector_to_cnn_vector(x);
    nng::Vector vgrad = do_compute_grad(v);
    nng::column_vector result = nng::cnn_vector_to_column_vector(vgrad);
    return result;
}
        

const nng::Vector nng::SparseAutoencoderGrad::do_compute_grad(nng::Vector& theta) const
{
    // Convert theta to the (W1, W2, b1, b2) matrix/scalar format
    // W1 = nng::Matrix2d(hidden_size,visible_size,theta.getSegment(0,hidden_size*visible_size));
    // W2 = nng::Matrix2d(visible_size,hidden_size,theta.getSegment(hidden_size*visible_size,hidden_size*visible_size));
    // b1 = theta(2*hidden_size*visible_size);
    // b2 = theta(2 * hidden_size * visible_size + hidden_size);

    nng::Matrix2d W1(hidden_size,visible_size,theta.getSegment(0,hidden_size*visible_size));
    nng::Matrix2d W2(visible_size,hidden_size,theta.getSegment(hidden_size*visible_size,hidden_size*visible_size));
    double b1 = theta(2 * hidden_size * visible_size);
    double b2 = theta(2 * hidden_size * visible_size + hidden_size);

    size_t m = data.get_cols(); // nb training examples

    // Forward propagation
     nng::Matrix2d z2 = W1*data + b1; // dim of z2: hidden_size*m
     nng::Matrix2d a2 = z2.sigmoid(); // dim of a2: hidden_size*m
     nng::Matrix2d z3 = W2*a2 + b2; // dim of z3 : visible_size*m
     nng::Matrix2d h = z3.sigmoid(); // dim of h: visible_size*m
    //forward_propagation(data);

    //compute_cost_and_backward_propagation(theta, data);
    // Sparsity
    nng::Vector vect_rho_hat = a2.sum(1) / m; //dim of rho_hat: hidden_size*1, average activation of hidden unit (average over training set)
    nng::Vector vect_rho = nng::Vector(hidden_size, sparsity_param);


    // Backprop
    nng::Matrix2d sparsity_delta_col(hidden_size, 1, ( -(vect_rho / vect_rho_hat) + (1 - vect_rho) / (1 - vect_rho_hat)) );
    nng::Matrix2d sparsity_delta(sparsity_delta_col); // dim sparsity_delta: hidden_size*m
    for (size_t i = 0; i < m-1; ++i)
    {
        sparsity_delta = sparsity_delta.concatenate(sparsity_delta_col,0);
    }


    nng::Matrix2d delta3 = -( data - h).dot(z3.sigmoid_prime()); // dJ/dz3. delta3: visible_size*m
    nng::Matrix2d delta2 = ((W2.transpose())*delta3 + beta * sparsity_delta).dot(z2.sigmoid_prime()); // dJ/dz2. dim delta2: hidden_size*m
    nng::Matrix2d W1grad = delta2*(data.transpose()) / m + lambda * W1;// dJ/dW1. dim W1grad: hidden_size*visible_size
    nng::Matrix2d W2grad = delta3*(a2.transpose()) / m + lambda * W2; // dJ/dW2. dim W2grad: visible_size*hidden_size
    nng::Vector b1grad = delta2.sum(1) / m; // dJ/db1. dim b1grad: hidden_size*1, sum over training set, because b1 was replicated for each trainig example, delta2 is the derivation of J over replicated b1 vector, should sum the influence of all the replicated b1 to obtain the derivation of J over a single b1
    nng::Vector b2grad = delta3.sum(1) / m; // dJ/db2. dim b2grad: visible_size*1, sum over training set, because b2 was replicated for each trainig example, delta3 is the derivation of J over replicated b2 vector, should sum the influence of all the replicated b2 to obtain the derivation of J over a single b2

    // After computing the cost and gradient, convert the gradients back
    // to a vector format (suitable for minFunc).  Specifically, we will unroll
    // gradient matrices into a vector.
	nng::Vector grad(hidden_size * visible_size * 2 + hidden_size + visible_size, 0);
    grad = W1grad.to_cnnvector().
            concatenate(W2grad.to_cnnvector()).
            concatenate(b1grad).
            concatenate(b2grad);
 
    return grad;
}
