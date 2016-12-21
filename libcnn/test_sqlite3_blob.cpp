#include "database.h"


int main()
{
	//double tabdouble= {0.1 , 0.2 , 0.3 , 0.4 , 0.005,0.00001 , 22.0};
	database db;
	db.open("test2.db");
	db.exec("CREATE TABLE IF NOT EXISTS filter (filter_id INTEGER PRIMARY KEY AUTOINCREMENT,lvl INTEGER , data BLOB);");
	db.exec("INSERT INTO filter (lvl) VALUES (3);");
	db.exec("INSERT INTO filter (lvl) VALUES (2);");
	//db.execute();
	char * buf = new char[100];
	memset(buf+90,0,1);
	memset(buf,'B',99);
	db.insert_blob("UPDATE filter SET data = (?) where lvl =1;",buf,94);
	int bsize=0;
	db.get_blob("SELECT data from filter where lvl =1",buf,bsize );
	//db.insert_blob("INSERT INTO filter (data) VALUES (?);",buf,94);
	
	db.open("tblob.db");
	db.addToDB();
	db.readFromDB();
	
	
	
}
