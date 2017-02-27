#include "cnn.h"
#include <math.h>
#include <iostream>

nng::Cnn::Cnn()
{}
    /*
    Returns the convolution of the features given by W and b with
    the given images
    :param patch_dim: patch (feature) dimension
    :param num_features: number of features
    :param images: large images to convolve with, matrix in the form
                   images(r, c, channel, image number)
    :param W: weights of the sparse autoencoder
    :param b: bias of the sparse autoencoder
    :param zca_white: zca whitening
    :param patch_mean: mean of the images
    :return:
    */
nng::Matrix4d nng::Cnn::cnnConvolve(size_t patch_dim, size_t num_features, const nng::Matrix4d& images, 
const nng::Matrix2d& W, const nng::Vector& b, const nng::Matrix2d& zca_white, const nng::Vector& patch_mean)
{

    size_t num_images = images.shape(3]);
    size_t image_dim = images.shape(0);
    size_t image_channels = images.shape(2);


    //    Convolve every feature with every large image to produce the
    //    numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1)
    //    matrix convolvedFeatures, such that
    //    convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
    //    value of the convolved featureNum feature for the imageNum image over
    //    the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)

    nng::Matrix4d convolved_features (num_features, num_images, image_dim - patch_dim + 1, image_dim - patch_dim + 1);
           
    nng::Matrix2d WT = W*zca_white; // dim W : hidden_size*visible_size, dim zca_white : visible_size*visible_size, dim WT : hidden_size*visible_size
    bT = b - WT*patch_mean; // dim b : hidden_size, dim patch_mean : visible_size, dim bT : hidden_size

	nng::Matrix2d convolved_image(image_dim - patch_dim + 1, image_dim - patch_dim + 1);
	nng::Matrix2d im(image_dim,image_dim);
	nng::Matrix2d feature(patch_dim, patch_dim);
	size_t patch_size = patch_dim * patch_dim;
    for (size_t i = 0; i < num_images; ++i)
	{
        for (size_t j = 0; j < num_features; ++j)
		{
            // convolution of image with feature matrix for each channel
            for (size_t channel = 0; channel < image_channels; ++channel)
			{
                // Obtain the feature (patchDim x patchDim) needed during the convolution
                //feature = WT[j, patch_size * channel:patch_size * (channel + 1)].reshape(patch_dim, patch_dim);
				feature = nng::Matrix2d(patch_dim, patch_dim, WT.getBlock(j,patch_size*channel,1,patch_size).toVector);

                // Flip the feature matrix because of the definition of convolution, as explained later
                //feature = np.flipud(np.fliplr(feature))

                // Obtain the image
                im = images[:, :, channel, i];

                // Convolve "feature" with "im", adding the result to convolvedImage
                convolved_image += nng::convolve2d(im, feature);
			}
            // Subtract the bias unit (correcting for the mean subtraction as well)
            // Then, apply the sigmoid function to get the hidden activation
            // convolved_image = sigmoid(convolved_image + bT[j]);
			convolved_image = convolved_image + bT[j];
			convolved_image = convolved_image.sigmoid();

            // The convolved feature is the sum of the convolved values for all channels
            convolved_features[j, i, :, :] = convolved_image;
		
		}
	}
    return convolved_features;
}

