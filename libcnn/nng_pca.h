#ifndef NNG_PCA_H
#define NNG_PCA_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"

#include "Vector.h"
#include "Matrix2d.h"


namespace nng
{

    class PCA_ZCA
    {
        public:
            PCA_ZCA(const Matrix2d& data, double _pca_ratio = 0.99, double regularization = 0.1, bool use_pca = false);
            ~PCA_ZCA(){};
            
            void setMean2Zero(); //set mean of each image data to zero
            void computeCovarianceMatrix();

            void whitening();
            
            const Matrix2d& getPcaWhiteData() const {return _data_pca_white;};
            const Matrix2d& getZcaWhiteData() const {return _data_zca_white;};
            const Matrix2d& getZcaWhite() const {return _zca_white;};
        private:
            nng::Matrix2d _data;
            double _pca_ratio;
            double _regularization;
            bool _use_pca;
            size_t _n; //size of each image vector
            size_t _m; //nb of images
            nng::Matrix2d _data_pca_white;
            nng::Matrix2d _data_zca_white;
            nng::Matrix2d _covariance;
            nng::Matrix2d _zca_white;

    };
}

#endif
