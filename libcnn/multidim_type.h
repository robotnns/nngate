#include <iostream>
#include <iostream>

template <class T>
class MultiDimMatrix
{
	public :
	MultiDimMatrix()
	{
			
	}
	MultiDimMatrix(u_long dimx,u_long dimy)
	{
		_vec = new T[dimx*dimy];
		_dimx=dimx;
		_dimy=dimy;
		_size=_dimx*_dimy;	
		dim=2;
	}
	
	MultiDimMatrix(u_long dimx,u_long dimy,u_long dimz)
	{
		_vec = new T[dimx*dimz];
		_dimx=dimx;
		_dimy=dimy;
		_dimz=dimz;		
	}
	
	
	
	
	T get2D(u_long x , u_long y )
	{
		
		return _vec[(_dimx*x)+y];
	}
	
	T set2D(u_long x , u_long y,T &&val )
	{
		
		 _vec[(_dimx*x)+y]=val;
	}

	MultiDimMatrix<T> add(MultiDimMatrix<T> & MatB)
	{
		//TODO check matric size input
		
		//T *_vecR = new T[_dimx*_dimy];		
		MultiDimMatrix<T> MatC(_dimx,_dimy);
		//CUDA ADD ->(_vec , Mat.getVec(),MatC.getVec())
		
		return MatC;
	}
	
	inline T * getVec()
	{
		return _vec;
	}
	
	void printVec()
	{
		 for (int i = 0 ; i < _size; i++)
		 { std::cout << _vec[i]<< ",";}
	 std::cout << std::endl;
		
	}
	
	T * _vec;
	u_long _dimx=0;
	u_long _dimy=0;
	u_long _dimz=0;
	u_long _dimw=0;
	u_long _size=0;
	u_short dim=0;
	
};