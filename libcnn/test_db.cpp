#include "database.h"


int main()
{
	//double tabdouble= {0.1 , 0.2 , 0.3 , 0.4 , 0.005,0.00001 , 22.0};
	database db;
	db.open("test2.db");
	db.exec("CREATE TABLE IF NOT EXISTS filter (filter_id INTEGER PRIMARY KEY AUTOINCREMENT,lvl INTEGER , data BLOB);");
	db.exec("INSERT INTO filter (lvl) VALUES (2);");
	db.exec("INSERT INTO filter (lvl) VALUES (2);");
	
	//db.execute();
	char * buf = new char[100];
	memset(buf+90,0,1);
	memset(buf,'B',99);
	db.insert_blob("UPDATE filter SET data = (?) where lvl =2;",buf,94);
	
	
	int bsize=0;
	bin_tuples bt;
	db.exec_bin("SELECT filter_id,data from filter where filter_id=1",bt,2 );

	std::cout << "number ligne " << bt.size() << std::endl;
		std::cout << "number col " << bt.at(0).size() << std::endl;
		std::cout << "val :" << bt.at(0).at(0).int_data<< std::endl;
		std::cout << "val :" << (char *)bt.at(0).at(1).blob_data<< std::endl;
	//db.insert_blob("INSERT INTO filter (data) VALUES (?);",buf,94);
	

	
	
}
