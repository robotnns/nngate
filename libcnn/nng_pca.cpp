#include "nng_pca.h"
#include "nng_math.h"
#include "nng_math_eig.h"

//each column in data is an image
nng::PCA_ZCA::PCA_ZCA(const Matrix2d& data, double pca_ratio, double regularization, bool use_pca):
_data(data)
,_pca_ratio(pca_ratio)
,_regularization(regularization)
,_use_pca(use_pca)
,_n(data.get_rows())
,_m(data.get_cols())
,_data_pca_white(data)
,_data_zca_white(data)
,_covariance(nng::Matrix2d(data.get_rows(),data.get_rows()))
,_zca_white(nng::Matrix2d(data.get_rows(),data.get_rows()))
{
    std::cout<<"pca zca"<<std::endl;
    setMean2Zero();
    computeCovarianceMatrix();
    whitening();
}

//set mean of each image data to zero
void nng::PCA_ZCA::setMean2Zero()
{
    std::cout<<"pca zca setMean2Zero"<<std::endl;
    double mean;
    nng::Vector img_data(_n);
    for (size_t i = 0; i < _m; ++i)
    {
        img_data = _data.get_col(i);
        mean = img_data.mean();
        img_data = img_data - mean;
        _data.set_col(img_data,i);
    }
}

void nng::PCA_ZCA::computeCovarianceMatrix()
{
    std::cout<<"pca zca computeCovarianceMatrix"<<std::endl;
    // nng::Matrix2d img_data(_n,1);
    // _covariance = nng::Matrix2d(_n,_n);
    // for (size_t i = 0; i < _m; ++i)
    // {
        // img_data = nng::Matrix2d(_n,1,_data.get_col(i));
        // _covariance += img_data * (img_data.transpose());
    // }
    // _covariance = _covariance/_m;
    _covariance = _data * _data.transpose();
    _covariance = _covariance/_m;
}



void nng::PCA_ZCA::whitening()
{
    std::cout<<"pca zca whitening"<<std::endl;
    
    nng::EigenValueEigenVector eig(_covariance);
    nng::Vector eigenvalue(eig.getEigenValue());
    nng::Matrix2d eigenvector(eig.getEigenVector());
    
    // compute number of components to retain
    double sum_eigvalue = eigenvalue.sum();
    double ratio = 0.0;
    size_t nb_components_to_retain = 0; //number of components to retain
    for (size_t i = 0; ratio<_pca_ratio && i < _n; ++i)
    {
        ratio += eigenvalue(i)/sum_eigvalue;
        nb_components_to_retain += 1;
    }
    std::cout<<"pca: original number of components = "<<_n<<std::endl;
    std::cout<<"pca: retained number of components = "<<nb_components_to_retain<<std::endl;
    
    nng::Vector regularized_eigenvalue(_n);
    regularized_eigenvalue = eigenvalue + _regularization;
    regularized_eigenvalue = 1.0 / regularized_eigenvalue;
    regularized_eigenvalue = regularized_eigenvalue.sqrt();
    
    if (_use_pca)
    {
        // PCA with dimension reduction with whitening and regularisation
        nng::Matrix2d img_rot_pca(nb_components_to_retain,_m);
        img_rot_pca = eigenvector.getBlock(0,0,_n,nb_components_to_retain).transpose() * _data; // rotate data
        nng::Matrix2d regularized_eigenvalue_pca = regularized_eigenvalue.getSegment(0,nb_components_to_retain).toDiagonal();
        _data_pca_white = eigenvector.getBlock(0,0,_n,nb_components_to_retain) * regularized_eigenvalue_pca * img_rot_pca;
    }
    
    // ZCA whitening
    _zca_white = eigenvector * regularized_eigenvalue.toDiagonal() * eigenvector.transpose();
    _data_zca_white = _zca_white * _data;
    
    // rotate data & whitening
    
    //_data_pca_white = nng::Matrix2d(nb_components_to_retain,_m);
    // nng::Matrix2d img_data(_n,1);
    // nng::Matrix2d img_rot(nb_components_to_retain,1);
    // nng::Vector img_white(nb_components_to_retain);
   
    // for (size_t i = 0; i < _m; ++i)
    // {
        // img_data = nng::Matrix2d(_n,1, _data.get_col(i));
        // // rotate
        // img_rot = eigenvector.getBlock(0,0,_n,nb_components_to_retain).transpose() * img_data;
        // // whitening
        // img_white = img_rot.to_cnnvector()/denom_regulation;
        // _data_pca_white.set_col(img_white,i);
    // }
    
    // _data_zca_white = eigenvector*_data_pca_white;
}
