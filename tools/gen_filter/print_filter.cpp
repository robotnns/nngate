#include <iostream>
#include <stdlib.h>
#include "filter_init_db.h"

int main(int argc , char ** argv)
{

 filter_init_db  fid;
 
    if(argc < 3)
     {
       std::cout << argv[0] << "<id_net> <filter_id (0 == not_used) > <filter_lvl  (-1 == notused) >\n";
       exit(0);
     }

    if(argv[2][0]!= '0')
    {
		std::cout << "s ID\n";
		fid.print_filter(argv[1],argv[2],NULL);
      
	}
	else 
	{
		if(strcmp(argv[3], "-1")!=0)
	 {
		 std::cout << "s LVL\n";
		fid.print_filter(argv[1],NULL,argv[3]);
		 
	 } 
	 else 
	 {
		 fid.print_filter(argv[1],NULL,NULL);
		 
	 }
		
	}
	 
}
