#ifndef PIX_DB_H
#define PIX_DB_H
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
class image_vec 
{
	public:
	image_vec(const int &height,const int &widths ,double &threshold,float &posx,float &posy, std::vector<double> *pix_vec)
	{
		_height = height;
		_widths = widths;
		_threshold=threshold;
		_pix_vec=pix_vec;
		_posx=posx;
		_posy=posy;
	}
	int _height;
	int _widths;
	float _posx=0;
	float _posy=0;
	
	double _threshold;
	std::vector<double> *_pix_vec;
};
class PixelMatrixViewer
{
	    
		public :
		void add_vec(const int &&height,const int &&widths ,double &&threshold,float &&posx,float &&posy,std::vector<double> *pix_vec)
		{
			auto iv =  image_vec(height,widths,threshold,posx,posy,pix_vec);
			v_im.push_back(iv);
		}
		void clear_vec()
		{
			v_im.clear();
			
		}
		
		
		std::vector<image_vec> v_im;
	void init(int wh,int wl)
	{
		 window.create(sf::VideoMode(wh, wl), "viewer");
		 window.setFramerateLimit(60);
		
	}
void draw_image_lib(int index)
{

    sf::Image        image;
	image.create(v_im.at(index)._widths, v_im.at(index)._height, sf::Color::Black );
    sf::Sprite       sprite;
    sf::Texture texture;
    sf::Uint8        *pixels  = new sf::Uint8[v_im.at(index)._widths*v_im.at(index)._height* 4];

	std::cout << "size:"<<v_im.at(index)._pix_vec->size() << std::endl;
       for (int y=0 ; y<v_im.at(index)._widths-1  ; y++)
		{
		for (int x=0 ; x<v_im.at(index)._height ; x++)
		 {
			std::cout << "x:"<< x << "y:" << y << std::endl;
			   double color  =v_im.at(index)._pix_vec->at(v_im.at(index)._widths*y+x);
		   /*if(color>v_im.at(index)._threshold)
			   image.setPixel (x,  y, sf::Color(255, 255, 255));
                else
                image.setPixel ( x,  y, sf::Color(0, 0, 255));*/
					//image.setPixel (x,y,grayToRgb((unsigned char)color));
				//std::cout << "COLOR = > " << 	color << std::endl;				
				image.setPixel (x,y,grayToRgb(color));
            }
			
			
        }

texture.loadFromImage(image);
sprite.setTexture(texture, true);
sprite.setPosition(v_im.at(index)._posx,v_im.at(index)._posy);
        window.draw(sprite);


}

sf::Color grayToRgb(double val)
{
	unsigned char ramapedVal = remapValue(val ,-1.0,1.0,0,255);
	return grayToRgb(ramapedVal);
	
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
};



#endif
