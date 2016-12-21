#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
#include <stdio.h>
#include "pixdb.h"
#define DATA_FOLDER "../../data/"

	sf::Font font;


void draw_text(std::string str, sf::RenderWindow &window,float x , float y,int size)
{

	sf::Text text;

// choix de la police à utiliser
text.setFont(font); // font est un sf::Font

// choix de la chaîne de caractères à afficher
text.setString(str.c_str());

// choix de la taille des caractères
text.setCharacterSize(size); // exprimée en pixels, pas en points !

text.setPosition(x,y);
// choix de la couleur du texte
text.setColor(sf::Color::Red);

// choix du style du texte
text.setStyle(sf::Text::Bold | sf::Text::Underlined);

// puis, dans la boucle de dessin, entre window.clear() et window.display()
window.draw(text);
}

void show_index(sf::RenderWindow &window,int cur_index , int total_size)
{
	std::string info = "";
	info =std::to_string(cur_index) +"/" +std::to_string(total_size);
	 draw_text(info,window,0,150,14);


}
void draw_pixel( sf::RenderWindow &window,float x , float y,char color)
{
	sf::RectangleShape rectangle(sf::Vector2f(1, 1));

	if(color==0)

	rectangle.setFillColor(sf::Color(255, 255, 255));
    else
	rectangle.setFillColor(sf::Color(0, 0, 255));

    rectangle.setPosition(x,y);
	window.draw(rectangle);
}

void draw_image(sf::RenderWindow &window,std::vector <STRU_PIXDB_REC> &v_rec,int index)
{

	for (int y=0 ; y<IMG_WIDTH_HEIGHT-1  ; y++)
		{
		for (int x=0 ; x<IMG_WIDTH_HEIGHT ; x++)
		 {
			 draw_pixel(window,x,y, v_rec.at(index).pix_buf[IMG_WIDTH_HEIGHT*y+x]);

		 }

		}

}
void draw_pixel2( float x , float y ,sf::Image & image,char color)
{


	if(color==0)

	image.setPixel(0, 0,sf::Color(255, 255, 255));
    else
	image.setPixel(0, 0,sf::Color(0, 0, 255));


}
void draw_image2(sf::RenderWindow &window,std::vector <STRU_PIXDB_REC> &v_rec,int index)
{

    sf::Image        image;
	image.create(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, sf::Color::Black );
    sf::Sprite       sprite;
    sf::Texture texture;
    sf::Uint8        *pixels  = new sf::Uint8[IMG_SIZE* 4];


       for (int y=0 ; y<IMG_WIDTH_HEIGHT-1  ; y++)
		{
		for (int x=0 ; x<IMG_WIDTH_HEIGHT ; x++)
		 {
			   char color =v_rec.at(index).pix_buf[IMG_WIDTH_HEIGHT*y+x];
		   if(color==0)
			   image.setPixel (x,  y, sf::Color(255, 255, 255));
                else
                image.setPixel ( x,  y, sf::Color(0, 0, 255));

            }
        }

texture.loadFromImage(image);
sprite.setTexture(texture, true);
        window.draw(sprite);


}

void show_fps(sf::RenderWindow &window,sf::Clock &clock, float &lastTime)
{

	 float currentTime = clock.restart().asSeconds();
            float fps =  1.f / currentTime;
            lastTime = currentTime;

			std::string info = "";
	info =std::to_string(fps) +" fps" ;
	 draw_text(info,window,0,164,14);


}

int main(int argc ,char  ** argv)
{
       if (argc <2)
	{std::cout << argv[0] << " <db filename>" << std::endl;
   		exit(0);}
    sf::RenderWindow window(sf::VideoMode(200, 200), "PIXDB viewer");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);
    pixdb pdb;
	pdb.set_file_name(argv[1]);
	std::vector <STRU_PIXDB_REC> v_rec;
	pdb.read_all(v_rec);
	int image_index=0;
	//glEnable(GL_TEXTURE_2D);
	window.setFramerateLimit(60);
	sf::Clock clock;
	float lastTime = 0;
if (!font.loadFromFile(DATA_FOLDER "Roboto-Black.ttf"))
{
    // erreur...
}

    while (window.isOpen())

    {
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
         {
					if(image_index>0)
				image_index--;
         }
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			{

				//v_rec
				if(image_index<v_rec.size()-1)
				image_index++;


			}

			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			{
               window.close();
            }


        window.clear();
		show_index(window,image_index,v_rec.size()-1);
		show_fps(window,clock,lastTime);


		  draw_image2(window,v_rec,image_index);
        window.display();
    }

    return 0;
}
