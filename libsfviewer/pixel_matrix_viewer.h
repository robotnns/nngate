#ifndef PIX_DB_H
#define PIX_DB_H
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
class image_vec 
{
	public:
	image_vec(const int &height,const int &widths ,float &posx,float &posy, std::vector<double> *pix_vec,float scale)
	{
		_height = height;
		_widths = widths;
		
		_pix_vec=pix_vec;
		_posx=posx;
		_posy=posy;
		_scale=scale;
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
		std::cout <<"min:" << _vec_min <<"max:"<<_vec_max<<std::endl;
	}
	int _height;
	int _widths;
	float _posx=0;
	float _posy=0;
	float _scale=1.0;
	double _vec_min;
	double _vec_max;
	
	
	std::vector<double> *_pix_vec;
};
class PixelMatrixViewer
{
	    
		public :
		void add_vec(const int &&height,const int &&widths ,float &&posx,float &&posy,std::vector<double> *pix_vec,float scale=1.0)
		{
			auto iv =  image_vec(height,widths,posx,posy,pix_vec,scale);
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

	
void draw_image_lib(int index)
{

    sf::Image        image;
	image.create(v_im.at(index)._widths, v_im.at(index)._height, sf::Color::Black );
    sf::Sprite       sprite;
    sf::Texture texture;
    sf::Uint8        *pixels  = new sf::Uint8[v_im.at(index)._widths*v_im.at(index)._height* 4];
    auto &img  = v_im.at(index);
	//std::cout << "size:"<<v_im.at(index)._pix_vec->size() << std::endl;
       for (int y=0 ; y<img._widths-1  ; y++)
		{
		for (int x=0 ; x<v_im.at(index)._height ; x++)
		 {
		//	std::cout << "x:"<< x << "y:" << y << std::endl;
			   double color  =img._pix_vec->at(img._widths*y+x);
		   /*if(color>v_im.at(index)._threshold)
			   image.setPixel (x,  y, sf::Color(255, 255, 255));
                else
                image.setPixel ( x,  y, sf::Color(0, 0, 255));*/
					//image.setPixel (x,y,grayToRgb((unsigned char)color));
				//std::cout << "COLOR = > " << 	color << std::endl;	
				if(_rgb_color)
				image.setPixel (x,y,grayToRgb(color,img._vec_min,img._vec_max));
				else
				image.setPixel (x,y,oneColorGreen(color,img._vec_min,img._vec_max));
            }
			
			
        }

	texture.loadFromImage(image);
	sprite.setTexture(texture, true);
	sprite.setPosition(img._posx,img._posy);
	if(img._scale!=1.0)
	sprite.setScale(img._scale,img._scale);
    window.draw(sprite);


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


float remapValue(double inputVal ,double minIn,double maxIn,double outMin,double outMax)
{
	
     return  (inputVal - minIn) * (outMax - outMin) / (maxIn - minIn) + outMin ;
}
void render()
{
	
	      
	  window.clear();
	  for (int i = 0 ; i <  v_im.size() ;i++ )
	 	 draw_image_lib(i);
      window.display();
	 
}
private:
sf::RenderWindow window;
bool _rgb_color=false;
};



#endif
