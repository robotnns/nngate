#ifndef PIX_MATIRX_VIEWER_H
#define PIX_MATIRX_VIEWER_H
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
class image_vec 
{
	public:
	image_vec(const int &height,const int &widths ,float &posx,float &posy, std::vector<double> *pix_vec,float scale,bool rgb_color=false)
	{
		_height = height;
		_widths = widths;
		
		_pix_vec=pix_vec;
		_posx=posx;
		_posy=posy;
		_scale=scale;
		_rgb_color=rgb_color;
		//find min max
		_vec_min=pix_vec->at(0);
		_vec_max=pix_vec->at(0);		
		for (auto &e : (*pix_vec))
		{
			if(e>_vec_max)
			_vec_max=e;
			if(e<_vec_min)
				_vec_min=e;
			
		}	
		//std::cout <<"min:" << _vec_min <<"max:"<<_vec_max<<std::endl;
		create_image();
	}
	bool save(std::string name  )
	{
		return _image.saveToFile (name);
	}
void create_image()	
{

  
	_image.create(_widths, _height, sf::Color::Black );
  
   
  /// sf::Uint8        *pixels  = new sf::Uint8[_widths*_height* 4];
  
	
       for (int y=0 ; y<_widths-1  ; y++)
		{
		for (int x=0 ; x<_height ; x++)
		 {		
			   double color  =_pix_vec->at(_widths*y+x);
				//std::cout << "color:" << color <<std::endl;
				if(_rgb_color)
				_image.setPixel (x,y,grayToRgb(color,_vec_min,_vec_max));
				else
				_image.setPixel (x,y,oneColorGreen(color,_vec_min,_vec_max));
            }
			
			
        }
		
	
	
}
void draw(sf::RenderWindow &window)
		{
			_texture.loadFromImage(_image);
	_sprite.setTexture(_texture, true);
	_sprite.setPosition(_posx,_posy);
	if(_scale!=1.0)
	_sprite.setScale(_scale,_scale);
			window.draw(_sprite);
		}
float remapValue(double inputVal ,double minIn,double maxIn,double outMin,double outMax)
{
	
     return  (inputVal - minIn) * (outMax - outMin) / (maxIn - minIn) + outMin ;
}

sf::Color grayToRgb(double val,double min , double max)
{
	unsigned char ramapedVal = remapValue(val ,min,max,0,255);
	return grayToRgb(ramapedVal);
	
}


sf::Color oneColorGreen(double val,double min , double max)
{
	unsigned char ramapedVal = remapValue(val ,min,max,0,255);
	return sf::Color(0,ramapedVal,0);
	
	
}
sf::Color grayToRgb(unsigned char Color )
{
	int Red   = (Color >> 5) * 255 / 7;
    int Green = ((Color >> 2) & 0x07) * 255 / 7;
    int Blue  = (Color & 0x03) * 255 / 3;
	return sf::Color(Red,Green,Blue);
}

	inline sf::Sprite  getSprite(){	return _sprite;	}
	
	int _height;
	int _widths;
	float _posx=0;
	float _posy=0;
	float _scale=1.0;
	double _vec_min;
	double _vec_max;
	bool _rgb_color;
	sf::Sprite   _sprite;
	sf::Image   _image;	
	sf::Texture _texture;
	std::vector<double> *_pix_vec;
	
	
   
};
class PixelMatrixViewer
{
	    
		public :
		void add_vec(const int &&height,const int &&widths ,float &&posx,float &&posy,std::vector<double> *pix_vec,float scale=1.0)
		{
			auto iv =  image_vec(height,widths,posx,posy,pix_vec,scale,_rgb_color);
			v_im.push_back(iv);
		}
		void clear_vec()
		{
			v_im.clear();
			
		}
		
		
		std::vector<image_vec> v_im;
	void init(int wh,int wl,bool rgb_color=false)
	{
		 window.create(sf::VideoMode(wh, wl), "viewer");
		 window.setFramerateLimit(60);
		 _rgb_color=rgb_color;
		
	}

void save_image(int index)
{
	v_im.at(index).save(std::to_string(index)+std::string(".png"));
}
void save_image(int index,std::string name)
{
	v_im.at(index).save(name);
}
void draw_image(int index)
{
   v_im.at(index).draw(window);
   
/*   auto &img  = v_im.at(index);
	window.draw(img.getSprite());*/
}



void render()
{
	
	      
	  window.clear();
	  for (unsigned int i = 0 ; i <  v_im.size() ;i++ )
	 	  draw_image(i);
      window.display();
	 
}
private:
sf::RenderWindow window;
bool _rgb_color=false;
};



#endif
