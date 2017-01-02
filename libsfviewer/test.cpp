#include "pixel_matrix_viewer.h"
#include <unistd.h>

int main()
{
	
	
	std::vector<double> pix_vec;
	for(int i = 0 ; i < 400;i++)
	pix_vec.push_back(2.4);
	PixelMatrixViewer pxv;
	std::cout << "before11"<<std::endl;
	
	/*float fcolor =pxv.remapValue(-0.22 ,-1.0,1.0,0,255);
	unsigned char color = fcolor;
	std::cout <<"f:" <<fcolor << "c:" <<(unsigned int)color<< std::endl;*/
	/*exit(0);
	*/
	pxv.init(200,250);
	pxv.add_vec(20,20,0.0,&pix_vec);
	std::cout << "before"<<std::endl;
	pxv.render();
	std::cout << "after"<<std::endl;
	sleep(4);
	
}
/*

#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
int main()
{
    // Create the main window
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML window");
    // Load a sprite to display
   
 
    while (window.isOpen())
    {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window : exit
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // Clear screen
        window.clear();
       
        // Update the window
        window.display();
    }
    return EXIT_SUCCESS;
}
*/