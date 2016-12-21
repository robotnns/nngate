#include <vector>
enum LayerType {conv=1,pooling=2,full=3};


class layer
{
	
	
	private :
	
	int filter_w;
	int filter_h;
	int filter_d;;
	int filter_stride;	
	int pading;
	double bias;
	
	std::vector <std::vector<double>> v_filter;
	std::vector <double> v_pixel;
	layer* prev;
	layer * next;
	
	
};


class net
{
	
	std::vector <layer> v_layer;
};


int main (int  argv , char ** argc)
{



return 0;
}
