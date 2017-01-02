#include "pixel_matrix_viewer.h"
#include <unistd.h>

int main()
{
	
	
	std::vector<double> pix_vec;
	for(int i = 0 ; i < 400;i++)
	pix_vec.push_back(0.224	);
	PixelMatrixViewer pxv;
	std::cout << "before11"<<std::endl;
	
	/*float fcolor =pxv.remapValue(0.22 ,-1.0,1.0,0,255);
	unsigned char color = fcolor;
	std::cout <<"f:" <<fcolor << "c:" <<(unsigned int)color<< std::endl;
	exit(0);*/
	
	pxv.init(800,600);
	pxv.add_vec(20,20,0.0,10.0,10.0,&pix_vec);
	pxv.add_vec(20,20,0.0,50.0,50.0,&pix_vec);
	pxv.add_vec(20,20,0.0,100.0,100.0,&pix_vec);
	pxv.add_vec(20,20,0.0,200.0,200.0,&pix_vec);
	pxv.render();	
	sleep(4);
	
}


