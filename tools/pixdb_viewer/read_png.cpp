#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#define REG_SIZE 2052
using namespace std;
#define RAW_PIX_SIZE 2016
/* FOREGROUND */
#define RST  "\x1B[0m"

#define KGRAY0  "\e[0;30m"
#define KGRAY1  "\e[0;37m"
//backgroundd
#define KGRAY01  "\e[0;40m"
#define KGRAY02  "\e[0;41m"
#define KGRAY03  "\e[0;42m"
#define KGRAY04  "\e[0;43m"
#define KGRAY05  "\e[0;44m"
#define KGRAY06  "\e[0;45m"
#define ROW_SIZE 128
#define PIXEL_BLOCK_SIZE 16523
#define PIXEL_FILTER_BLOCK_SIZE 1512
#define PIXEL_SIZE 16384

struct IDAT_STR
{
	
	short zlib_header;     //Zlib header 0x78 0x01
	unsigned char b;       //The byte 0x00 or 0x01, depending on if it is a middle or the last block.
	short num_byte_block ; //Number of bytes in block (up to 2^16-1) stored as a little endian 16-bit integer
	short one_comp ;       //The 1's complement of this integer representation.
	char pix_buf[PIXEL_BLOCK_SIZE] ; // pixel + filter byte
	int zlib_checksum;  // An adler-32 checksum of all the image data
};

void draw_pixel(short lvl)
{
	char cc= '@';
	
	if( lvl == 0)
		std::cout << KGRAY05        <<cc <<  RST;
	 
	else 
		std::cout << KGRAY06      <<cc << RST;
	
}



int main (int argc , char ** argv)
{



	int width = 128		;
	int height = 129;
     streampos size;
  char * memblock;

  ifstream file(argv[1] , ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    int nbr_enreg = size / REG_SIZE ;

    std::cout << "nombre d'enreg:" << nbr_enreg << std::endl;
    memblock = new char [size];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();


   
    //delete[] memblock;
  }else cout << "Unable to open file \n";
  

  
  char pix_buf[PIXEL_SIZE];
  memset(pix_buf,2,PIXEL_SIZE);

  
  //filter all bad data
  for (int i=7 , j=0 , row_size =0; i<PIXEL_BLOCK_SIZE-4  ; i++) 
  {
			if(row_size!=0)
			{
			 pix_buf[j]=memblock[i]; // pixel + filter byte
			 j++;
			 
			 if(row_size==ROW_SIZE)
			 {
				 row_size=0;
			 }
			}
			row_size++;
			
		
  }
  
  int line_size=0;
		for (int x=0 ; x<width ; x++) 
		 {
			  line_size=x;
		 }
			 std::cout <<"line offset :"<< line_size<<std::endl;

  for (int y=0 ; y<height-1  ; y++) 
		{
		for (int x=0 ; x<width ; x++) 
		 {
			 			
			draw_pixel(pix_buf[height*y+x]);
		 }
		  std::cout <<"|"<<y;
		   std::cout <<std::endl;
		 
		}
		
  

		


  return 0;
}
