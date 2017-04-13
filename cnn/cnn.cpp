#include "cnn.h"
#include <math.h>
#include <iostream>

nng::Cnn::Cnn(size_t patch_dim, size_t num_features, nng::Matrix4d& data)
:_patch_dim(patch_dim)
,_num_features(num_features)
,_images(data)
,_image_dim(data.shape()[0])
,_num_images(data.shape()[3])
,_image_channels(data.shape()[2])
,_convolved_features(nng::Matrix4d(_image_dim - _patch_dim + 1, _image_dim - _patch_dim + 1,_num_features, _num_images))
{}
    /*
    Returns the convolution of the features given by W and b with
    the given images
    :param patch_dim: patch (feature) dimension
    :param num_features: number of features
    :param images: large images to convolve with, matrix in the form
                   images(r, c, channel, image number)
    :param W: weights of the sparse autoencoder
			  rows of W corresponds to each feature, nb of rows = hidden_size 
			  cols of W is organized as feature value for channel 1, feature value for channel 2, ..., nb of cols = visible_size
    :param b: bias of the sparse autoencoder
    :param zca_white: zca whitening
    :param patch_mean: mean of the images
    :return:
    */
nng::Matrix4d nng::Cnn::cnnConvolve(const nng::Matrix2d& W, const nng::Vector& b, const nng::Matrix2d& zca_white, const nng::Vector& patch_mean)
{

    //size_t num_images = images.shape(3);
    //size_t image_dim = images.shape(0);
    //size_t image_channels = images.shape(2);


    //    Convolve every feature with every large image to produce the
    //    numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1)
    //    matrix convolvedFeatures, such that
    //    convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
    //    value of the convolved featureNum feature for the imageNum image over
    //    the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)

    //nng::Matrix4d convolved_features (num_features, num_images, image_dim - patch_dim + 1, image_dim - patch_dim + 1);
	//nng::Matrix4d convolved_features (image_dim - _patch_dim + 1, image_dim - _patch_dim + 1,_num_features, num_images);
           
    nng::Matrix2d WT = W*zca_white; // dim W : hidden_size*visible_size, dim zca_white : visible_size*visible_size, dim WT : hidden_size*visible_size
    nng::Vector bT = b - WT*patch_mean; // dim b : hidden_size, dim patch_mean : visible_size, dim bT : hidden_size

	nng::Matrix2d convolved_image(_image_dim - _patch_dim + 1, _image_dim - _patch_dim + 1);
	nng::Matrix2d im(_image_dim,_image_dim);
	nng::Matrix2d feature(_patch_dim, _patch_dim);
	size_t patch_size = _patch_dim * _patch_dim;
    for (size_t i = 0; i < _num_images; ++i)
	{
        for (size_t j = 0; j < _num_features; ++j)
		{
            // convolution of image with feature matrix for each channel
            for (size_t channel = 0; channel < _image_channels; ++channel)
			{
                // Obtain the feature (patchDim x patchDim) needed during the convolution
                //feature = WT[j, patch_size * channel:patch_size * (channel + 1)].reshape(patch_dim, patch_dim);
				feature = nng::Matrix2d(_patch_dim, _patch_dim, WT.getBlock(j,patch_size*channel,1,patch_size).toVector);

                // Flip the feature matrix because of the definition of convolution, as explained later
                //feature = np.flipud(np.fliplr(feature))

                // Obtain the image
                //im = images[:, :, channel, i];
				im = _images.getMatrix2d(channel,i);

                // Convolve "feature" with "im", adding the result to convolvedImage
                convolved_image += nng::convolve2d(im, feature);
			}
            // Subtract the bias unit (correcting for the mean subtraction as well)
            // Then, apply the sigmoid function to get the hidden activation
            // convolved_image = sigmoid(convolved_image + bT[j]);
			convolved_image = convolved_image + bT[j];
			convolved_image = convolved_image.sigmoid();

            // The convolved feature is the sum of the convolved values for all channels
            //convolved_features[j, i, :, :] = convolved_image;
			_convolved_features.setMatrix2d(convolved_image,j,i);
		
		}
	}
    return _convolved_features;
}

