#ifndef CNN_H
#define CNN_H

#include "nng_math.h"
#include "dlib/matrix/matrix.h"


namespace nng{
    
    class Cnn
    {
        public:
            Cnn(size_t patch_dim, size_t num_features, nng::Matrix4d& data);
            ~Cnn();
			Matrix4d cnnConvolve(const Matrix2d& W, const Vector& b, const Matrix2d& zca_white, const Vector& patch_mean);
 
		private:
			size_t _patch_dim;
			size_t _num_features;
			Matrix4d _images;
			size_t _image_dim;
			size_t _num_images;
			size_t _image_channels;
			Matrix4d _convolved_features;



    };

}
#endif
