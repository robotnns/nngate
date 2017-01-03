#include "pixel_matrix_viewer.h"
#include <unistd.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
double frand_a_b(double a, double b){

    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}
int main()
{
	
	 srand (time(NULL));
	std::vector<double> pix_vec;
	for(int i = 0 ; i < 800;i++)
	pix_vec.push_back(frand_a_b(-1.0,1.0)	);


	PixelMatrixViewer pxv;
	
	
	/*float fcolor =pxv.remapValue(0.22 ,-1.0,1.0,0,255);
	unsigned char color = fcolor;
	std::cout <<"f:" <<fcolor << "c:" <<(unsigned int)color<< std::endl;
	exit(0);*/
	
	pxv.init(800,600);
	pxv.add_vec(20,20,10.0,10.0,&pix_vec,2.0);
	pxv.add_vec(20,20,50.0,50.0,&pix_vec);
	pxv.render();	
	sleep(4);
	
}


