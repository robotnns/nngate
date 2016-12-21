#ifndef PIX_DB_H
#define PIX_DB_H
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

#define IDAT "IDAT"
#define IHDR "IHDR"
#define IEND "IEND"
#define PIXEL_FILTER_BLOCK_SIZE 16512
#define IDAT_DATA_SIZE 16523
#define PNG_SIG_SIZE 8
#define SCNALINE_SIZE 129
#define IHDR_SIZE 25
#define IMG_WIDTH_HEIGHT 128
#define IMG_SIZE IMG_WIDTH_HEIGHT*IMG_WIDTH_HEIGHT

using namespace std;

struct STRU_PIXDB_HEADER
{

	int record_count;


};

struct STRU_PIXDB_REC
{

	//int recnum;     //Zlib header 0x78 0x01
	char label;
	char  pix_buf[IMG_SIZE];
};


class pixdb
{
	public :
	void set_file_name(std::string filename);
	void open(char * filename);
    int read_count();	
	void write_db_entry(char * pix_buf,char label);
	bool read_all(std::vector <STRU_PIXDB_REC> &v_rec);

	private :

	ifstream ifile;
	ofstream ofile;
	std::string m_filename="o.pdb";

};
#endif
