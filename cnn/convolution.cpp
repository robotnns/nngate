#include <iostream> 
#include <vector>
//suppose image size is Wx,Hx 
// filter size is Wf,Hf
// stride is s

// F is one filter
bool convolution(std::vector<double>& X, std::vector<double>& F, int s, int Wx, int Hx, int Wf, int Hf, std::vector<double>& output)
{
    output.clear();

    if (Wx*Hx != X.size())
	{
		std::cout<<"error in data size"<<std::endl;
		return false;
	}
    
    if (Wf*Hf != F.size())
	{
		std::cout<<"error in filter size"<<std::endl;
		return false;
	}
    
	if ((Wx-Wf)%s != 0 || (Hx-Hf)%s != 0)
	{
		std::cout<<"error: stride value not compatible with data and filter size"<<std::endl;
		return false;
	}
	
	for (int i=0;i<=(Wx-Wf);i+=s)
	{
		for (int j=0;j<=(Hx-Hf);j+=s)
		{
			double sum = 0;
			for (int l=0;l<Wf;l++)
			{
				for (int k=0;k<Hf;k++)
				{	
					sum += X[Wx*(i+l)+j+k]*F[Wf*l+k];
				}
			}
			output.push_back(sum);
		
		}
	}
	return true;
}

bool max_pooling(std::vector<double>& X, int s, int Wx, int Hx, int Wf, int Hf, std::vector<double>& output)
{
    output.clear();
    
    if (Wx*Hx != X.size())
	{
		std::cout<<"error in data size"<<std::endl;
		return false;
	}
    
    
	if ((Wx-Wf)%s != 0 || (Hx-Hf)%s != 0)
	{
		std::cout<<"error: stride value not compatible with data and filter size"<<std::endl;
		return false;
	}
    
	for (int i=0;i<=(Wx-Wf);i+=s)
	{
		for (int j=0;j<=(Hx-Hf);j+=s)
		{
			double max = X[Wx*i+j];
			for (int l=0;l<Wf;l++)
			{
				for (int k=0;k<Hf;k++)
				{	
                    if (X[Wx*(i+l)+j+k] > max)
                        max = X[Wx*(i+l)+j+k];
				}
			}
			output.push_back(max);
		
		}
	}
	return true;
}

// F is a vector of all the single weights in a fully-connected layer
bool fully_connected(std::vector<double>& X, std::vector<double>& weights, int Wx, int Hx, std::vector<double>& output)
{
    output.clear();
    for (std::vector<double>::iterator it = weights.begin() ; it != weights.end(); ++it)
    {
        std::vector<double> F(Wx*Hx, *it);
        std::vector<double> output_i;
        convolution(X,F,1,Wx,Hx,Wx,Hx,output_i);
        output.push_back(output_i[0]);
    }
	return true;
}

int main()
{	
	std::vector<double> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	std::vector<double> filter = {7, 5, 16, 8};
	std::vector<double> output;
	bool result;
    int stride = 1;
    int width_data = 4;
    int height_data = 4;
    int witdh_filter = 2;
    int height_filter = 2;
    //stride = 1
    std::cout<<"--------test conv 1--------"<<std::endl;
    result = convolution(data,filter,stride,width_data,height_data,witdh_filter,height_filter,output);
	std::cout<<"result="<<result<<std::endl;
	for (int i=0;i<output.size();++i)
		std::cout<<output[i]<<" ";
	std::cout<<std::endl;

    //correct result
	std::cout<<"result2"<<std::endl;
	double o1,o2,o3,o4;
    o1 = data[0]*filter[0]+data[1]*filter[1]+data[4]*filter[2]+data[5]*filter[3];//1*7+2*5+5*16+6*8;
	o2 = data[1]*filter[0]+data[2]*filter[1]+data[5]*filter[2]+data[6]*filter[3];//2*7+3*5+6*16+7*8;
	o3 = data[2]*filter[0]+data[3]*filter[1]+data[6]*filter[2]+data[7]*filter[3];//3*7+4*5+7*16+8*8;
	o4 = data[4]*filter[0]+data[5]*filter[1]+data[8]*filter[2]+data[9]*filter[3];//5*7+6*5+9*16+10*8;
	std::cout<<o1<<" "<<o2<<" "<<o3<<" "<<o4<<"..."<<std::endl;

	//stride = 2
    std::cout<<"--------test conv 2--------"<<std::endl;
    stride = 2;
	result = convolution(data,filter,stride,width_data,height_data,witdh_filter,height_filter,output);
	std::cout<<"result="<<result<<std::endl;
	for (int i=0;i<output.size();++i)
		std::cout<<output[i]<<" ";
	std::cout<<std::endl;

	//stride = 3 should not work
    std::cout<<"--------test conv 3--------"<<std::endl;
    stride = 3;
	result = convolution(data,filter,stride,width_data,height_data,witdh_filter,height_filter,output);

    // test max pooling
    std::cout<<"--------test max pooling--------"<<std::endl;
    result = max_pooling(data,1,width_data,height_data,witdh_filter,height_filter,output);
	std::cout<<"result="<<result<<std::endl;
	for (int i=0;i<output.size();++i)
		std::cout<<output[i]<<" ";
	std::cout<<std::endl;
    
    // test fully connected
    std::cout<<"--------test fully connected layer--------"<<std::endl;
    std::vector<double> weights = {1,2};
    result = fully_connected(data,weights,width_data,height_data,output);
	std::cout<<"result="<<result<<std::endl;
	for (int i=0;i<output.size();++i)
		std::cout<<output[i]<<" ";
	std::cout<<std::endl;
}
