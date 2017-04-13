#include "nng_math.h"
#include "nng_math_eig.h"
#include <iostream>
#include <vector>
#include "nng_pca.h"
#include <pixdb.h>
#include <pixel_matrix_viewer.h>

using namespace std;

void testMatrix2d()
{
  nng::Matrix2d mat1(4, 4, 1.0);
  nng::Matrix2d mat2(4, 4, 2.0);
  cout<<"mat1="<<endl;
  mat1.print();

  cout<<"mat2="<<endl;
  mat2.print();
	
  cout<<"test Matrix/Matrix operations"<<endl;
  cout<<"mat3 = mat1"<<endl;
  nng::Matrix2d mat3=mat1;
  mat3.print();

  cout<<"mat3 = mat1 + mat2;"<<endl;
  mat3 = mat1 + mat2;
  mat3.print();

  cout<<"mat3 += mat1;"<<endl;
  mat3 += mat1;
  mat3.print();

  cout<<" mat3 = mat2 - mat1;"<<endl;
  mat3 = mat2 - mat1;
  mat3.print();

  cout<<" mat3-= mat1;"<<endl;
  mat3-= mat1;
  mat3.print();

  cout<<" mat3 = mat1*mat2;"<<endl;
  mat3 = mat1*mat2;
  mat3.print();

  cout<<" mat3 *= mat1;"<<endl;
  mat3 *= mat1;
  mat3.print();

  cout<<" -mat3;"<<endl;
  nng::Matrix2d m = -mat3;
  m.print();
	
  cout<<" mat3.dot(mat2);"<<endl;
  mat3.dot(mat2).print();

	
  cout<<"test Matrix/scalar operations"<<endl;
	
  cout<<" mat3 = mat1 + 2;"<<endl;
  mat3 = mat1 + 2;
  mat3.print();
  cout<<" mat6 = 2 + mat1;"<<endl;
  nng::Matrix2d mat6 = 2 + mat1;
  mat6.print();
	
  cout<<" mat3 = mat1 - 2;"<<endl;
  mat3 = mat1 - 2;
  mat3.print();
  cout<<" mat6 = 2 - mat1;"<<endl;
  mat6 = 2 - mat1;
  mat6.print();
	
  cout<<" mat3 = mat1 * 2;"<<endl;
  mat3 = mat1 * 2;
  mat3.print();
  cout<<" mat6 = 2 * mat1;"<<endl;
  mat6 = 2 * mat1;
  mat6.print();
	
  cout<<" mat3 =mat1 / 2;"<<endl;
  mat3 = mat1 / 2;
  mat3.print();

  cout<<"test Matrix/vector operations"<<endl;
 nng::Vectord v1(4,0);
  v1[0] = 1;
  v1[1] = 2;
  v1[2] = 3;
  v1[3] = 4;
  cout<<"v1="<<endl;
  nng::print_v(v1);

  nng::Vectord v2(4,0);
  cout<<"  v2 = mat1*v1;"<<endl;
  v2 = mat1*v1;
  nng::print_v(v2);

  cout<<"test transpose"<<endl;
  mat1(0,0) = 1;
  mat1(0,1) = 2;
  mat1(0,2) = 3;
  mat1(0,3) = 4;
  cout<<"mat1="<<endl;
  mat1.print();

  cout<<"mat3 = mat1.transpose()"<<endl;
  mat3 = mat1.transpose();
  mat3.print();

  cout<<"test block operations"<<endl;
  cout<<"mat3.getBlock(1,0,2,2)"<<endl;
  mat3.getBlock(1,0,2,2).print();

  nng::Matrix2d mat4(2, 2, 5.0);
  cout<<"mat4="<<endl;	
  mat4.print();
  cout<<" mat3.setBlock(0,1,2,2, mat4)"<<endl;
  mat3.setBlock(0,1,2,2, mat4);
  mat3.print();

  cout <<"mat3.argmax(0)"<<endl;
  mat3.argmax(0).print();
	
  cout<<"test concatenation"<<endl;
  cout<<"mat1.concatenate(mat2,0)"<<endl;
  mat1.concatenate(mat2,0).print();
  cout<<"mat1.concatenate(mat2,1)"<<endl;
  mat1.concatenate(mat2,1).print();

  cout<<"mat1.sigmoid()"<<endl;
  mat1.sigmoid().print();
	
  cout<<"mat1.sum(0)"<<endl;
  mat1.sum(0).print();
	
  cout<<"mat1.sum(1)"<<endl;
  mat1.sum(1).print();
	
  cout<<"mat1.sum()"<<endl;
  cout<<mat1.sum()<<endl;
	
  cout<<"mat1.power(2)"<<endl;
  mat1.power(2).print();
	
  cout<<"mat1.sigmoid_prime()"<<endl;
  mat1.sigmoid_prime().print();
	
  cout<<"mat1.max()"<<endl;
  cout<<mat1.max()<<endl;	
	
  cout<<"mat1.exp()"<<endl;
  mat1.exp().print();
	
  cout<<"test vectord to matrix conversion"<<endl;
  cout<<"mat5(2,2,v1)"<<endl;
  nng::Matrix2d mat5(2,2,v1);
  mat5.print();
  cout<<"test nng::Vector to matrix conversion"<<endl;
  nng::Vector v3(v1);
  nng::Matrix2d mat8(2,2,v3);
  mat8.print();
	
  cout<<"test Matrix to vectord conversion"<<endl;
  cout<<"v3 = mat5.toVector()"<<endl;
  nng::Vectord v4 = mat5.toVector();
  nng::print_v(v4);

  cout<<"test Matrix to nng::Vector conversion"<<endl;
  cout<<"mat5.to_cnnvector()"<<endl;
  mat5.to_cnnvector().print();

}

void testCnnVector()
{
	cout<<" nng::Vector v1(4,1)"<<endl;
    nng::Vector v1(4,1);
    v1.print();

    nng::Vectord v(4,2);
	cout<<"v="<<endl;
	nng::print_v(v);
	
	cout<<" nng::Vector v2(nng::Vectord v)"<<endl;
    nng::Vector v2(v);
    v2.print();

	cout<<" nng::Vector v3(nng::Vector v2)"<<endl;
    nng::Vector v3(v2);
    v3.print();

	cout<<"v1(2)"<<endl;
	cout<<v1(2)<<endl;
	
	cout<<"v1[2]"<<endl;
	cout<<v1[2]<<endl;
	
	cout<<" v4 = v1.concatenate(nng::Vectord v)"<<endl;
    nng::Vector v4 = v1.concatenate(v);
    v4.print();

	cout<<" v5 = v1.concatenate(nng::Vector v2)"<<endl;
    nng::Vector v5 = v1.concatenate(v2);
    v5.print();

	cout<<" v6 = v1+2"<<endl;
    nng::Vector v6 = v1+2;
    v6.print();
	cout<<" v7 = 2 + v1"<<endl;
    nng::Vector v7 = 2 + v1;
    v7.print();
	
	cout<<" v6 = v1-2"<<endl;
    v6 = v1-2;
    v6.print();
	cout<<" v7 = 2 - v1"<<endl;
    v7 = 2 - v1;
    v7.print();
	
	cout<<" v6 = v1*2"<<endl;
    v6 = v1*2;
    v6.print();
	cout<<" v7 = 2 * v1"<<endl;
    v7 = 2 * v1;
    v7.print();
	
	cout<<" v6 = v1/2"<<endl;
    v6 = v1/2;
    v6.print();
	
    cout<<" v8 = v4.getSegment(2,4)"<<endl;
    nng::Vector v8 = v4.getSegment(2,4);
    v8.print();

	cout<<" v4.setSegment(2,4,v1)"<<endl;
    v4.setSegment(2,4,v1);
	v4.print();
	
	cout<<"v3.kl_divergence(v7)"<<endl;
	v3.kl_divergence(v7).print();
	
	cout<<"v3.kl_divergence(v8)"<<endl;
	v3.kl_divergence(v8).print();

	cout<<"v4.sum()"<<endl;
	cout<<v4.sum()<<endl;
	
	cout<<"v4.dot(v4)"<<endl;
	v4.dot(v4).print();
	
	cout<<"v4.getVector()"<<endl;
	nng::print_v(v4.getVector());
	
}

void testStandardNormalDistrubutionGenerator()
{
  cout<<"standard normal distribution generator"<<endl;
  cout<<nng::normal_distribution_rand(0.0,1.0)<<endl;	
}

void testEigenValueEigenVector()
{
//	nng::Matrix2d mat(2, 2);
//	mat(0,0) = 2;
//	mat(0,1) = 1;
//	mat(1,0) = 1;
//	mat(1,1) = 2;
	nng::Matrix2d mat(3, 3);
	mat(0,0) = 1;mat(0,1) = 0;mat(0,2) = 0;
	mat(1,0) = 0;mat(1,1) = 2;mat(1,2) = 0;
	mat(2,0) = 0;mat(2,1) = 0;mat(2,2) = 3;
	nng::EigenValueEigenVector eig(mat);
	nng::Vector eig_value(eig.getEigenValue());
	nng::Matrix2d eig_vector(eig.getEigenVector());
	eig_value.print();
	eig_vector.print();
    int n = mat.get_cols();
	nng::Vector result1(n);
	nng::Vector result2(n);
    for(int i = 0; i < n;++i)
    {
        result1 = mat*eig_vector.get_col(i);
        result2 = eig_vector.get_col(i);
        result2 = result2*eig_value(i);
        result1.print();
        result2.print();
    }


}

void testPcaZca()
{
	std::vector <STRU_PIXDB_REC_DOUBLE> v_image_in;
	pixdb pdb;
	pdb.set_file_name("../../data/G.pdb");	
	pdb.read_all(v_image_in);
    size_t patch_size = 28;
    size_t input_size = patch_size*patch_size;
    nng::Matrix2d data(input_size,1);
    nng::Matrix2d data_white(input_size,1);
	size_t i_start_x, i_start_y;
	nng::Matrix2d m_image(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT);
	nng::Matrix2d m_patch(patch_size, patch_size,0);
	i_start_x = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
	i_start_y = (size_t)(nng::rand_a_b(40.0, 1.0*(IMG_WIDTH_HEIGHT - patch_size-40.0)));
    m_image = nng::Matrix2d(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, v_image_in[0].pix_buf);
	m_patch = m_image.getBlock(i_start_x, i_start_y, patch_size, patch_size);
	data.set_col(m_patch.toVector(), 0);
    
    nng::PCA_ZCA zca(data,0.99,0.1);
    data_white = zca.getZcaWhite();
    
    std::vector<double> raw_image = data.get_col(0);
    std::vector<double> zca_white_image = data_white.get_col(0);
    
	PixelMatrixViewer pxv;
	pxv.init(1000,1000,false);

	float scale = 4.0;
    pxv.add_vec(IMG_WIDTH_HEIGHT,IMG_WIDTH_HEIGHT,0,0,&raw_image,scale);
    pxv.add_vec(IMG_WIDTH_HEIGHT,IMG_WIDTH_HEIGHT,IMG_WIDTH_HEIGHT*scale,0,&zca_white_image,scale);
	
	pxv.render();	
	pxv.save_image(0);
	sleep(100);
}

void testMatrix4d()
{
	nng::Matrix4d m1(2,2,2,3);
	m1.print();
	
	nng::Matrix4d m2(2,2,2,3,1.0);
	m2.print();
	
	nng::Vector v(24);
	for (int i=0;i<24;++i)
		v[i] = i;
		
	nng::Matrix4d m3(2,2,2,3,v);
	m3.print();
	
	nng::Vectord v2 = v.getVector();
	nng::Matrix4d m4(2,2,2,3,v2);
	m4.print();
	
	nng::Matrix4d m5(m3);
	m5.print();
	
	nng::Matrix4d m6 = m5;
	m6.print();
	
	std::cout<<m6(1,1,1,2)<<std::endl;
	
	std::vector<size_t> shape = m6.shape();
	for(auto s:shape)
		std::cout<<s<<std::endl;
	
	nng::Matrix2d m7 = m6.getMatrix2d(1,2);
	m7.print();
	
	nng::Matrix2d m8 = m7+1;
	m6.setMatrix2d(m8,1,2);
	m6.print();
	
}

void testConvolution()
{
	std::vector<double> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	std::vector<double> filter = {7, 5, 16, 8};
	std::vector<double> output;

    size_t stride = 1;
    size_t width_data = 4;
    size_t height_data = 4;
    size_t width_filter = 2;
    size_t height_filter = 2;
	
	nng::Matrix2d im(width_data, height_data, data);
	nng::Matrix2d feature(width_filter, height_filter, filter);
    //stride = 1
    std::cout<<"--------test conv 1--------"<<std::endl;
    nng::Matrix2d result = nng::convolve2d(im,feature,stride);
	result.print();

    //correct result
	std::cout<<"reference result"<<std::endl;
	double o1,o2,o3,o4;
    o1 = data[0]*filter[0]+data[1]*filter[1]+data[4]*filter[2]+data[5]*filter[3];//1*7+2*5+5*16+6*8;
	o2 = data[1]*filter[0]+data[2]*filter[1]+data[5]*filter[2]+data[6]*filter[3];//2*7+3*5+6*16+7*8;
	o3 = data[2]*filter[0]+data[3]*filter[1]+data[6]*filter[2]+data[7]*filter[3];//3*7+4*5+7*16+8*8;
	o4 = data[4]*filter[0]+data[5]*filter[1]+data[8]*filter[2]+data[9]*filter[3];//5*7+6*5+9*16+10*8;
	std::cout<<o1<<" "<<o2<<" "<<o3<<" "<<o4<<"..."<<std::endl;	
}

int main(int argc, char **argv) {
  //testMatrix2d();
  //testCnnVector();
  //testStandardNormalDistrubutionGenerator();
  //testEigenValueEigenVector();
  //testPcaZca();//TODO
  //testMatrix4d();
  testConvolution();
  return 0;
}
