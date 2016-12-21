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

void ech_color()
{
	
	
	std::cout <<
	KGRAY01 <<"@" <<RST
	<< KGRAY02<<"@"<<RST
	<< KGRAY03<<"@"<<RST
	<< KGRAY04<<"@"<<RST
	<<KGRAY05	<<"@"<<RST
	<< RST<<std::endl;
}

inline short swap16(short val) {
    return ((val & 0xFF) << 8)
           | ((val >> 8) & 0xFF);
}
#pragma pack(push, 1) 

struct etl_rec{
	
	short data_num;
	short label_ascii;
	short serial_sheet_number;
	unsigned char jis_code;
	unsigned char ebcdic;
	unsigned char eval_individual_char ;
	unsigned char eval_group_char ;
	unsigned char male_female;
	unsigned char age_of_writer;
	int 	      serial_num;
	short         industry_class_code;
	short         occupation_class_code;
	short         sheet_gather_date_yy_mm;
	short         scanning_date_yy_mm;
	unsigned char y_sample_pos;
	unsigned char x_sample_pos;
	unsigned char min_scan_lvl;
	unsigned char max_scan_lvl;
	short          undef1;
	short          undef2;
	char           raw_pixel[RAW_PIX_SIZE];  //4 bits by pix (16 gray lvl)
	int            uncertain;
};
#pragma pack(pop)

typedef std::vector<etl_rec> V_ETL_REC;

void show_stat(char * memblock,streampos size)
{
	    char c   = 0;
    int count = 0;
	for (int i = 0 ; i < size ; i+=REG_SIZE )
    {
      if (c != memblock[i+2])
       {
           cout << count <<" samples -> " << c << std::endl;
	   count =0;
	   c = memblock[i+2];
       }      
       else 
         count++;
     
    }
	
}

void parse_data(char * memblock,streampos size,V_ETL_REC &v_rec)
{
	
	struct etl_rec rec;
	for (int i = 0 ; i < size ; i+=sizeof(etl_rec)  )
    {
		memset(&rec,0,sizeof(etl_rec) );
		memcpy(&rec, &memblock[i],sizeof(etl_rec)  );
		v_rec.push_back(rec);
		//std::cout  << "add data" << rec.data_num << std::endl;
	}
	

}

int load_data_etl(char * filename,V_ETL_REC &v_rec)
{
	
 // std::cout << "size struct : " << sizeof(etl_rec)  << std::endl;
  std::cout << "FILE : " <<  filename << std::endl;
  streampos size;
  char * memblock;

  ifstream file(filename, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    int nbr_enreg = size / REG_SIZE ;

    std::cout << "nombre d'enreg:" << nbr_enreg << std::endl;
    memblock = new char [size];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();


//show_stat(memblock,size);
    parse_data (memblock,size,v_rec);
    //delete[] memblock;
  }
  

  else cout << "Unable to open file \n";
	
}
void draw_pixel(short lvl)
{
	char cc= '@';
	
	if( lvl == 0)
		std::cout << KGRAY06        <<lvl <<  RST;
	 else if(lvl == 1)
			std::cout << KGRAY01        <<lvl <<  RST;
    else if(lvl == 2)
		std::cout << KGRAY02      <<lvl << RST;
	 else if(lvl == 3)
		std::cout << KGRAY03      <<lvl << RST;
	 else if(lvl == 4)
		std::cout << KGRAY04      <<lvl << RST;
	else if(lvl > 4)
		std::cout << KGRAY05      <<lvl << RST;
	
}
int main (int argc , char ** argv)
{
	ech_color();
	V_ETL_REC v_rec;
	load_data_etl( argv[1] , v_rec);
	//for (auto r : v_rec)
	//{
		//std::cout  << "n:" << swap16(r.data_num)<< std::endl;
		//std::cout  << "n:" << swap16(r.data_num)<< std::endl;
		//std::cout  << "a:" << (short)r.age_of_writer<< std::endl;
		
		
	//}
	int ind=atoi(argv[2]);
	std::cout << "label: " << (char) v_rec.at(ind).label_ascii << std::endl;
 	// Specify an output image size
	int width = 64	;
	int height = 63;
    //short gray_lvl[64][63];
	short gray_lvl[RAW_PIX_SIZE*2+1];
	short i_gray=0;
	for (int i = 0 ; i  <RAW_PIX_SIZE ; i++ ) 
	{
		
			short b1=(short) ((v_rec.at(ind).raw_pixel[i]&0x0F));
			short b2=(short) ((v_rec.at(ind).raw_pixel[i]&0xF0)>>4);
			gray_lvl[i_gray]=b2;
			gray_lvl[i_gray+1]=b1;
			i_gray+=2;
			/*gray_lvl[x][y]=b1;
			gray_lvl[x][y]=b2;*/
		    
			
			
		
	} 
		
		for (int y=0 ; y<height  ; y++) 
		{
		for (int x=0 ; x<width ; x++) 
		 {
			draw_pixel(gray_lvl[height*y+x]);
		 }
		   std::cout <<std::endl;
		}
			
		
		


  return 0;
}
