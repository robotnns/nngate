#ifndef H_DATABASE
#define H_DATABASE
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include "debug.h"
#include <vector>
#include <map>
#include <iostream>
#define DEFAULT_DATABASE_PATH "../mb_common/mbox.db"
#include <string.h>

struct struct_bin_rec
{
	int data_type;
	int data_size;
	void * blob_data;
	const unsigned char * str_data;
	int int_data;
	double double_data;


};

typedef std::map<std::string, std::string> m_row;
typedef std::vector<m_row>  v_tuples;
typedef std::vector<std::vector<struct struct_bin_rec>>  bin_tuples;



class database
{
    public :
    int open();
	int exec_bin(const std::string &req , bin_tuples &bt,int column_count);
    int exec_bin(const char * req , bin_tuples &bt,int column_count);
    int open (const char * file_name);
	int open (const std::string &file_name);
    int exec (const char *q);
	int exec (const std::string &q);
    int open_exec(const char *q);
	 int open_exec(const std::string &q);
	int insert_blob(const std::string &req ,void *data,int blob_size );
	int get_blob(const std::string &req ,void *data,int &blob_size );
    void readFromDB();
    void addToDB();
    void insert_blob();
    void close();
    v_tuples get_tuple();
    v_tuples get_tuples();
    v_tuples tuples;
template<typename T, typename... Args>
void test_VA(T &data ,Args... args) {
   data++;
   test_VA(args...);
}

	private:
	//sqlite3_stmt *_statement;
    sqlite3 *handle;
};
#endif
