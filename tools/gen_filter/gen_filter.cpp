#include <iostream>
#include <stdlib.h>
#include "filter_init_db.h"
/*double frand_a_b(double a, double b){

    return (( rand()/(double)RAND_MAX ) * (b-a) + a);
}

int main()
{
int n ;
 srand (time(NULL));

double min = -0.02;
double max = 0.02;

for (int i= 0 ; i < 100 ; i ++ )
{
double res = frand_a_b(min,max)	;
std::cout << res  << std::endl;
}
return 0 ;
}*/

void test()
{
	database db;
    db.open("test_filter.db");
	 db.exec("delete filter ;" );
	db.exec("CREATE TABLE IF NOT EXISTS filter (filter_id INTEGER PRIMARY KEY AUTOINCREMENT,lvl INTEGER , data BLOB);");
    db.exec("INSERT INTO filter (lvl) VALUES (0);" );
	//db.insert_blob("UPDATE filter SET data = (?) where filter_id =(select MAX(filter_id) from filter );",(void*)"ABCDEFGH",9);
	double *dd=new double[3];
	dd[0]=0.2214;
	dd[1]=0.2216;
	dd[2]=0.2219;
    db.insert_blob("UPDATE filter SET data = (?) where filter_id =(select MAX(filter_id) from filter );",(void*)dd,sizeof(double)*3);		
	char *dat;
	 bin_tuples bt;
	db.exec_bin("SELECT data from filter",bt,1 );
	//dat =(char*) bt.at(0).at(0).blob_data;
	 //std::cout <<" data : " << dat << std::endl;
	 double * dd2 =(double*)bt.at(0).at(0).blob_data; 
	 std::cout <<" data : " << dd2[0]  << std::endl;
	 std::cout <<" data : " << dd2[1]  << std::endl;
	 std::cout <<" data : " << dd2[2]  << std::endl;
	
	exit(0);
	
	
}
int main(int argc , char ** argv)
{
//	test();
 filter_init_db  fid;
 //fid.print_filter("1",-1);
    if(argc < 3)
     {
      std::cout << argv[0] << " <generate filter net_id (0 == not caculate filter) > <print filter id_net (0 == no print filter) >\n";
    exit(0);
     }

    if(argv[1][0]!= '0')
    {
      std::cout << "generate filter in progress ..."<<std::endl;
	 fid.init_net_filter(argv[1]);
	}
	 if(argv[2][0]!= '0')
	  fid.print_filter(argv[2],NULL,NULL);
}
