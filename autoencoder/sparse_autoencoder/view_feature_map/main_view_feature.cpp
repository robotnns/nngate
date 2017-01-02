#include "nng_math.h"
#include <iostream>
#include <fstream>

struct FeatureMapData
{
	nng::Matrix2d* feature_map;
	size_t image_numbers;
	size_t image_width;
	size_t image_height;
};

std::vector<std::string> split(const std::string &s, char delim) 
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (std::getline(ss, item, delim)) 
	{
        tokens.push_back(item);
    }
    return tokens;
}

FeatureMapData read_autoencoder_feature_map(const char* filename);
void draw(std::vector<double> data, size_t width, size_t height);


int main(int argc, char **argv) 
{
	FeatureMapData feature_map_data = read_autoencoder_feature_map(argv[1]);
	nng::Matrix2d feature_map = *feature_map_data.feature_map;
	size_t image_numbers = feature_map_data.image_numbers;
	size_t width = feature_map_data.image_width;
	size_t height = feature_map_data.image_height;
	
	
	feature_map = feature_map - feature_map.min();
	double feature_map_max = feature_map.abs().max();
	feature_map = feature_map / feature_map_max;
	
	feature_map = feature_map * 255;
	feature_map = feature_map.floor();
	

	//std::vector<double> data = feature_map.get_row(0);
	//std::vector<int> image_data(data.begin(), data.end());
	//nng::print_v(feature_map.get_row(0));
	for (size_t i = 0; i < image_numbers; i++)
		draw(feature_map.get_row(i), width, height);
	
	return 0;
}	

FeatureMapData read_autoencoder_feature_map(const char* filename)
{
  std::cout << "Opening autoencoder feature map file"<<std::endl;
  std::string line;
  std::ifstream file (filename);
  size_t hidden_size = 0;
  size_t input_size = 0;
  size_t patch_size = 0;
  
  FeatureMapData feature_map_data;
  
  if (file.is_open())
  {
	std::getline (file,line);
	std::getline (file,line);
	hidden_size = std::stoi(line);
	std::cout << "hidden_size = "<< hidden_size << std::endl;
	std::getline (file,line);
	std::getline (file,line);
	input_size = std::stoi(line);
	std::cout << "input_size = "<< input_size << std::endl;
	std::getline (file,line);
	std::getline (file,line);
	patch_size = std::stoi(line);
	std::cout << "patch_size = "<< patch_size << std::endl;
	
	std::getline (file,line);  
	std::vector<std::string>   data_str = split(line, ' ');
	size_t total_size = patch_size*patch_size*hidden_size;
	assert(total_size == data_str.size());
	
	nng::Matrix2d feature(hidden_size, input_size,0);
	for (size_t i = 0; i < hidden_size; i++)
	{
		for (size_t j = 0; j < input_size; j++)
			feature(i,j) = std::stod(data_str[i*input_size + j]);
	}
	
    file.close();
	feature_map_data.feature_map = new nng::Matrix2d(feature);

  }

  else
  { 
	  std::cout << "Unable to open autoencoder feature file"<<std::endl; 	
	  nng::Matrix2d feature(hidden_size, input_size,0);
	  feature_map_data.feature_map = new nng::Matrix2d(feature);
  }
  
	feature_map_data.image_numbers = hidden_size;
	feature_map_data.image_width = patch_size;
	feature_map_data.image_height = patch_size;

	return feature_map_data;  
}

void draw(std::vector<double> data, size_t width, size_t height)
{
	std::vector<int> image_data(data.begin(), data.end()); 
}

