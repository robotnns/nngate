#include "filter_init_db.h"
#define ALL_FILTER_LVL_ -1
#include <fstream>


double filter_init_db::rand_a_b(double a, double b)
{

    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}

void filter_init_db::fill_random_tab(double * tab ,unsigned long size,double min , double max)
{
     for(unsigned long  i = 0 ; i < size ; i ++ )
     {
          tab[i] = rand_a_b(min,max);

     }


}

void filter_init_db::print_double_tab(double * tab ,unsigned long size)
{

	std::cout << "==  double table ==" <<std::endl;
     for(unsigned long  i = 0 ; i < size ; i ++ )
     {
          std::cout << "[" << i << "] ->" << tab[i] <<std::endl;

     }


}
void  filter_init_db::init_lvl(const std::string & lvl , int filter_count , int filter_size,double min,double max, database & db)
 {

	for(int i=0 ; i < filter_count ; i++)
	{
	  //double * rand_dtab = new double [filter_size];
	 double  rand_dtab[filter_size];
      fill_random_tab(rand_dtab,filter_size,min,max);

#ifdef DEBUG_PRINT
	    std::cout << "B filter num " << i << std::endl;
	  print_double_tab(rand_dtab,filter_size);
	 // print_double_tab(rand_dtab,filter_size);
#endif
	  db.exec("INSERT INTO filter (lvl) VALUES ("+ lvl +");" );
	  db.insert_blob("UPDATE filter SET data = (?) where filter_id =(select MAX(filter_id) from filter where lvl="+lvl+");",(void*)rand_dtab,filter_size*sizeof(double));
     //delete rand_dtab;
	 
        std::cout<<"lvl:" << lvl <<  " filters: " << i<<"/"<< filter_count<<"\r"<< std::flush;
     //std::cout << "." << std::flush;

	}

 }


void  filter_init_db::print_filter(const std::string id_net,char * filter_id,char * lvl)
 {
 database db_filter;
    db_filter.open(id_net +"_filter.db");
	 bin_tuples bt;
	 std::string req = ""; 
	 if(filter_id != NULL)
	 {
		 req="SELECT data from filter where filter_id ="+std::string(filter_id);
		 
		 
	 }
	 else
	 {
		if (lvl == NULL)
		{
			req="SELECT data from filter ;";
			
		}
		else
			req="SELECT data from filter where lvl ="+std::string(lvl);
			
	 }
	 std::cout<<"EXECUTE " << req<< std::endl;
    db_filter.exec_bin(req,bt,1 );
	for(unsigned i=0 ; i < bt.size() ; i++)
	{

			struct struct_bin_rec brec = bt.at(i).at(0);
		    std::cout << "A filter num " << i << std::endl;

			print_double_tab((double*)brec.blob_data,brec.data_size /sizeof(double));
	}

db_filter.close();
 }
void filter_init_db::init_net_filter(const std::string id_net)
 {
v_tuples tuple_param;
	database db_filter;
	database net_pdb;
	db_filter.open(id_net +"_filter.db");
	net_pdb.open(_param_db_path);
	net_pdb.exec("select * from layer where fk_id_net =" + id_net );
       tuple_param = net_pdb.get_tuples();
       db_filter.exec("CREATE TABLE IF NOT EXISTS filter (filter_id INTEGER PRIMARY KEY AUTOINCREMENT,lvl INTEGER , data BLOB);");

       for( auto row : tuple_param)
	{

           init_lvl(row["layer_level"],std::stoi(row["filter_count"]),
		   std::stoi(row["filter_height"])*std::stoi(row["filter_width"]),
		   std::stod(row["filter_min_val"]),
		   std::stod(row["filter_max_val"]),
		   db_filter );
        }
	//db.execute();
	/*char * buf = new char[100];
	memset(buf+90,0,1);
	memset(buf,'B',99);
	int bsize=0;
	db.get_blob("SELECT data from filter where lvl =1",buf,bsize );
	//db.insert_blob("INSERT INTO filter (data) VALUES (?);",buf,94);

	db.open("tblob.db");
	db.addToDB();
	db.readFromDB();*/
	db_filter.close();
net_pdb.close();


 }
