
#include "pixdb.h"
	void pixdb::set_file_name(std::string filename)
	{
         m_filename = filename;
	}

	void pixdb::open(char * filename)
	{


	}
        int pixdb::read_count()
        {
			return 0;
        }
	void pixdb::write_db_entry(char * pix_buf,char label)
	{


        struct STRU_PIXDB_REC pixdb_rec;

      ofile.open(m_filename,std::ofstream::binary | std::ostream::app);
       if (ofile.is_open())
       {

      pixdb_rec.label =label;
        memcpy(&pixdb_rec.pix_buf,pix_buf,IMG_WIDTH_HEIGHT*IMG_WIDTH_HEIGHT);
	    ofile.write((char *)&pixdb_rec, sizeof(struct STRU_PIXDB_REC));
	    ofile.flush();
	    ofile.close();

	   }
	   else {std::cout << "file not open:" << m_filename << std::endl; }
	}

	bool pixdb::read_all(std::vector <STRU_PIXDB_REC> &v_rec)
	{
		 ifile.open(m_filename,ios_base::binary);
	 if (ifile.is_open())
       {
		   ifile.seekg (0, ios::end);
           int  file_size = ifile.tellg();
           ifile.seekg (0, ios::beg);
			int nbr_it=file_size/IMG_SIZE;
	    struct STRU_PIXDB_REC pixdb_rec;
	   // ifile.read((char *)&pixdb_rec,  sizeof(struct STRU_PIXDB_REC));
	   
	   //char tab[sizeof(struct STRU_PIXDB_REC)];
	   for(int i = 0 ; i < nbr_it ; i+=IMG_SIZE+1)
	   {
				ifile.read((char *)&pixdb_rec,  sizeof(struct STRU_PIXDB_REC));
				 v_rec.push_back(pixdb_rec);
				  ifile.seekg (i, ios::beg);
	   } 
   
		
        }

	   else {std::cout << "file not open:" << m_filename << std::endl; }


	   ifile.close();
	   return true;
	}


