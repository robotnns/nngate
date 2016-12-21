#include "cnn_util.h"
#include <iostream>
#include <vector>
using namespace std;
using namespace util;
void testMatrix2d()
{
  Matrix2d mat1(4, 4, 1.0);
  Matrix2d mat2(4, 4, 2.0);
  cout<<"mat1="<<endl;
  mat1.print();

  cout<<"mat2="<<endl;
  mat2.print();
	
  cout<<"test Matrix/Matrix operations"<<endl;
  cout<<"mat3 = mat1"<<endl;
  Matrix2d mat3=mat1;
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
  Matrix2d m = -mat3;
  m.print();
	
  cout<<" mat3.dot(mat2);"<<endl;
  mat3.dot(mat2).print();

	
  cout<<"test Matrix/scalar operations"<<endl;
	
  cout<<" mat3 = mat1 + 2;"<<endl;
  mat3 = mat1 + 2;
  mat3.print();
  cout<<" mat6 = 2 + mat1;"<<endl;
  Matrix2d mat6 = 2 + mat1;
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
 Vectord v1(4,0);
  v1[0] = 1;
  v1[1] = 2;
  v1[2] = 3;
  v1[3] = 4;
  cout<<"v1="<<endl;
  util::print_v(v1);

  Vectord v2(4,0);
  cout<<"  v2 = mat1*v1;"<<endl;
  v2 = mat1*v1;
  util::print_v(v2);

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

  Matrix2d mat4(2, 2, 5.0);
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
  Matrix2d mat5(2,2,v1);
  mat5.print();
  cout<<"test CnnVector to matrix conversion"<<endl;
  CnnVector v3(v1);
  Matrix2d mat8(2,2,v3);
  mat8.print();
	
  cout<<"test Matrix to vectord conversion"<<endl;
  cout<<"v3 = mat5.toVector()"<<endl;
  Vectord v4 = mat5.toVector();
  util::print_v(v4);

  cout<<"test Matrix to CnnVector conversion"<<endl;
  cout<<"mat5.to_cnnvector()"<<endl;
  mat5.to_cnnvector().print();

}

void testCnnVector()
{
	cout<<" CnnVector v1(4,1)"<<endl;
    CnnVector v1(4,1);
    v1.print();

    Vectord v(4,2);
	cout<<"v="<<endl;
	util::print_v(v);
	
	cout<<" CnnVector v2(Vectord v)"<<endl;
    CnnVector v2(v);
    v2.print();

	cout<<" CnnVector v3(CnnVector v2)"<<endl;
    CnnVector v3(v2);
    v3.print();

	cout<<"v1(2)"<<endl;
	cout<<v1(2)<<endl;
	
	cout<<"v1[2]"<<endl;
	cout<<v1[2]<<endl;
	
	cout<<" v4 = v1.concatenate(Vectord v)"<<endl;
    CnnVector v4 = v1.concatenate(v);
    v4.print();

	cout<<" v5 = v1.concatenate(CnnVector v2)"<<endl;
    CnnVector v5 = v1.concatenate(v2);
    v5.print();

	cout<<" v6 = v1+2"<<endl;
    CnnVector v6 = v1+2;
    v6.print();
	cout<<" v7 = 2 + v1"<<endl;
    CnnVector v7 = 2 + v1;
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
    CnnVector v8 = v4.getSegment(2,4);
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
	util::print_v(v4.getVector());
	
}

int main(int argc, char **argv) {
  testMatrix2d();
  testCnnVector();
	
  cout<<"standard normal distribution generator"<<endl;
  cout<<util::normal_distribution_rand(0.0,1.0)<<endl;
  return 0;
}
