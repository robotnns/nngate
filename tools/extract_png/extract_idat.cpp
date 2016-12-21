#include <iostream>
#include <fstream>
#include <string.h>
#include <fstream>
#include <vector>


#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>


using namespace std;

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
#pragma pack(push, 1)

#include "pixdb.h"

class png_list
 {

 public:

 png_list(std::string folder)
 {

	 m_folder = folder;
 }



 void build()
 {
	 affdossier(m_folder.c_str());

 }
 std::vector<std::string> get_list()
{
	return v_pnglist;

}

 private:
void affdossier (const char *nom_dossier) {

DIR *dossier;
struct dirent *contenu_dossier;
struct stat infos;
printf("Affichage de %s : \n", nom_dossier);
if((dossier = opendir(nom_dossier)) != NULL)
{
        chdir(nom_dossier);

		char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) == NULL)
		{exit(3);}
        while ((contenu_dossier = readdir(dossier)) != NULL)

        if ( (strcmp(".", contenu_dossier->d_name) != 0)
        && (strcmp("..", contenu_dossier->d_name) != 0) )
        {
                stat(contenu_dossier->d_name, &infos);
                if(S_ISDIR(infos.st_mode))
                        affdossier(contenu_dossier->d_name);


                else
				{
						//check if png file extension
					   int fsize=strlen(contenu_dossier->d_name);
					   if(contenu_dossier->d_name[fsize-3]=='p' && 
					       contenu_dossier->d_name[fsize-2]=='n' &&
						   contenu_dossier->d_name[fsize-1]=='g')
				       v_pnglist.push_back(std::string(cwd)+std::string("/")+std::string(contenu_dossier->d_name) );
				 }
				//printf(" %s %s \n",cwd, contenu_dossier->d_name);


		}

chdir("..");



}


        if (closedir(dossier) == -1) {
        perror("Fermeture du repertoire");
        exit(2);
        }

}

private :
std::string m_folder;
std::vector<std::string> v_pnglist;

 };

 int filecount=0;
void affdossier (char *nom_dossier) {

DIR *dossier;
struct dirent *contenu_dossier;
struct stat infos;
printf("Affichage de %s : \n", nom_dossier);
if((dossier = opendir(nom_dossier)) != NULL)
{
        chdir(nom_dossier);
        while ((contenu_dossier = readdir(dossier)) != NULL)

        if ( (strcmp(".", contenu_dossier->d_name) != 0)
        && (strcmp("..", contenu_dossier->d_name) != 0) )
        {
                stat(contenu_dossier->d_name, &infos);
                if(S_ISDIR(infos.st_mode))
                        affdossier(contenu_dossier->d_name);


                else
				{
					filecount++;
					printf(" %s \n", contenu_dossier->d_name);

				}
		}

chdir("..");



}


        if (closedir(dossier) == -1) {
        perror("Fermeture du repertoire");
        exit(2);
        }

}



struct STRU_IDAT_DATA_ZONE
{

	short zlib_header;     //Zlib header 0x78 0x01
	unsigned char b;       //The byte 0x00 or 0x01, depending on if it is a middle or the last block.
	short num_byte_block ; //Number of bytes in block (up to 2^16-1) stored as a little endian 16-bit integer
	short one_comp ;       //The 1's complement of this integer representation.
	char pix_buf[PIXEL_FILTER_BLOCK_SIZE] ; // pixel + filter byte
	int zlib_checksum;  // An adler-32 checksum of all the image data
};

struct STRU_IDAT_CHUNK
{
	int lenght ;
	char chunk_type[4] ;
	STRU_IDAT_DATA_ZONE data_zone;
	unsigned char crc[4]  ;

};

#pragma pack(pop)

inline short swap16(short val) {
    return ((val & 0xFF) << 8)
           | ((val >> 8) & 0xFF);
}

inline int swap32(int x)
{
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF));
}

int load_data_idat(std::string filename,struct STRU_IDAT_CHUNK &idat)
{

 // std::cout << "size struct : " << sizeof(etl_rec)  << std::endl;
  std::cout << "FILE : " <<  filename << std::endl;
  streampos size;
  char * memblock;
  ifstream file(filename, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();

    memblock = new char [size];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();


	 int idat_size=0;


	 memset(&idat,0,sizeof(struct STRU_IDAT_CHUNK));
	 memcpy(&idat,memblock+PNG_SIG_SIZE+IHDR_SIZE,sizeof(struct STRU_IDAT_CHUNK));
	 idat.lenght=swap32(idat.lenght);
	 if(IDAT_DATA_SIZE != idat.lenght)
	 {

		 std::cout << "ce fichier n'es pas conforme IDAT_DATA_SIZE:"<<IDAT_DATA_SIZE << "!=" << "idat_size" << idat.lenght <<"not match"<< std::endl;
		 exit(0);
	 }





  }
	 else std::cout << "cannot open file:"<< filename << std::endl;


 }

void write_pixel_file(const char * filename , char * pixel_buf,int size)
{
	fstream outfile;

	try {

    outfile.open(filename, ios::out | ios::binary);
    outfile.write(pixel_buf,size);

    outfile.close();
	}
    catch(...){
		std::cout<< "write_pixel_file() exeption on open file";

	}
//
	//std::cout << fileout << " writed "  << std::endl;



}
void check(struct STRU_IDAT_CHUNK idat)
{


	std::cout <<"size :" <<idat.lenght

	<< std::endl;
		printf("zlib header:%x \n",idat.data_zone.zlib_header);

	for(int i = 0 ; i < PIXEL_FILTER_BLOCK_SIZE ; i+=(SCNALINE_SIZE)  )
	{
		int a = idat.data_zone.pix_buf[i];
     	printf("byte[%d]:%d \n",i,a);
	}



}

void extract_pixel(struct  STRU_IDAT_CHUNK &idat,char * pixel_buf)
{


     memset(pixel_buf,0,IMG_WIDTH_HEIGHT*IMG_WIDTH_HEIGHT);
	for(int i = 0, j=0; i < PIXEL_FILTER_BLOCK_SIZE ; i+=(SCNALINE_SIZE) ,j+=IMG_WIDTH_HEIGHT )
	{
		memcpy(pixel_buf+j,&idat.data_zone.pix_buf[i+1],IMG_WIDTH_HEIGHT);
		//std::cout <<i << "\n";
	}



}


void draw_pixel(short lvl)
{
	char cc= '@';

	if( lvl == 0)
		std::cout << KGRAY05        <<cc <<  RST;

	else
		std::cout << KGRAY06      <<cc << RST;

}

void draw_pix_buff(char * pixel_buf)
{

	 for (int y=0 ; y<IMG_WIDTH_HEIGHT-1  ; y++)
		{
		for (int x=0 ; x<IMG_WIDTH_HEIGHT ; x++)
		 {

			draw_pixel(pixel_buf[IMG_WIDTH_HEIGHT*y+x]);
		 }
		  std::cout <<"|"<<y;
		   std::cout <<std::endl;

		}




}

void process_png(std::string  png_filename,char label,char delete_file_flag)
{
	std::string png_filename_decomp =png_filename + ".decomp";
	std::string dbname ="";
	dbname.push_back(label);
	dbname+=".pdb";

	std::string cmd="pngcrush -force -c 0 -m 1 -l 0 " + png_filename  + " " + png_filename_decomp +" > /dev/null 2>&1 ";
	std::cout << "execute :" << cmd << std::endl;
 	system (cmd.c_str());

	struct STRU_IDAT_CHUNK idat;
	char * pixel_buf=new char [IMG_WIDTH_HEIGHT*IMG_WIDTH_HEIGHT];
	load_data_idat(png_filename_decomp,idat);
	//check(idat);
	extract_pixel(idat,pixel_buf);
	//write_pixel_file("out.pix",pixel_buf,IMG_WIDTH_HEIGHT*IMG_WIDTH_HEIGHT);
	//std::cout << pixel_buf[0]  << pixel_buf[1] << std::endl;
	//draw_pix_buff(pixel_buf);
	try{
	pixdb pdb;
	pdb.set_file_name(dbname);
	pdb.write_db_entry(pixel_buf,label);
	//delete if flag set
	if( delete_file_flag == '1' )
	unlink(png_filename_decomp.c_str());
}
    catch(...){
		std::cout<< "exeption on open file";

	}

}
int main (int argc , char ** argv)
{
	char flag_delete_decomp_file='0';
     if (argc < 3 )
	 {
		 std::cout << "usage :" << argv[0] <<  " <input folder of png> < label > <delete .decomp file flag [delete:'1' notdelete:'0'] >  " <<std::endl;
		 exit(-1);
	 }
	 if (argc > 3 )
	 	   flag_delete_decomp_file = argv[3][0];
	   

		
png_list plist(argv[1]);
plist.build();
auto pl = plist.get_list();

std::cout << "nbr files to process: " << pl.size() << std::endl;
for (auto r : pl )
{
	process_png( r,argv[2][0],flag_delete_decomp_file);
	//printf(" file => %s \n", r.c_str());
}

}
