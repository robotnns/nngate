#include "database.h"






int  database::insert_blob(const std::string &req ,void *data,int blob_size )
{
   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    int res = SQLITE_ERROR;
	sqlite3_stmt *updStmt =NULL;

    res = sqlite3_prepare_v2(handle,  req.c_str(), -1, &updStmt, NULL);

	if(res!= SQLITE_OK)
    {
	DBG_ERROR("sqlite3_prepare_v2  error\n");
	return SQLITE_ERROR;

    }

	//res = sqlite3_bind_blob(updStmt, 1, data, blob_size , SQLITE_TRANSIENT);
	res = sqlite3_bind_blob(updStmt, 1, data, blob_size ,SQLITE_STATIC);
 if((res = sqlite3_step(updStmt)) != SQLITE_DONE)
    {
		DBG_ERROR("sqlite3_step  error\n");
        sqlite3_reset(updStmt);
		return SQLITE_ERROR;

	}
	res = sqlite3_reset(updStmt);
	sqlite3_finalize(updStmt);
	return SQLITE_DONE;
}

int database::exec_bin(const std::string &req , bin_tuples &bt,int column_count)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
	return exec_bin( req.c_str() , bt, column_count);
}
int database::exec_bin(const char * req , bin_tuples &bt,int column_count)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    int res = SQLITE_ERROR;
    sqlite3_stmt *statement;

    int  field_size;

    res = sqlite3_prepare_v2 (handle,(const char *) req , -1, &statement, NULL);

    if (res == SQLITE_OK)
    {
		//int count_step = 0 ;
        while (sqlite3_step(statement) == SQLITE_ROW)
        {
            std::vector<struct struct_bin_rec> v_sbr;
            for(int clmn = 0 ; clmn < column_count ; clmn++)
             {
                field_size = sqlite3_column_bytes(statement, clmn);
				//std::cout << "f size -> " << field_size << std::endl;
                if(field_size>0)
                {
                    int data_type =  sqlite3_column_type(statement, clmn);
                    /*
#define SQLITE_INTEGER  1
#define SQLITE_FLOAT    2
#define SQLITE_BLOB     4
#define SQLITE_NULL     5
#ifdef SQLITE_TEXT
# undef SQLITE_TEXT
#else
# define SQLITE_TEXT     3
#endif
#define SQLITE3_TEXT     3a*/
                   struct struct_bin_rec br ;



                   switch(data_type)
                   {
                     case  SQLITE_BLOB :{
					 // br.data = malloc(field_size);
// std::cout << "SQLITE_BLOB " << std::endl;
data_type= SQLITE_BLOB;

                    void * data_blob =(void *)sqlite3_column_blob(statement, clmn) ;
                    br.blob_data = malloc(field_size);
                   memcpy(br.blob_data ,data_blob,field_size);
                   break;
                   }
                     case SQLITE_TEXT :
					 //std::cout << "SQLITE_TEXT " << std::endl;
					 data_type= SQLITE_TEXT;
                   br.str_data =sqlite3_column_text(statement, clmn) ;
                   break;
                     case SQLITE_INTEGER:
					 data_type= SQLITE_INTEGER;
	 //td::cout << "SQLITE_INTEGER " << std::endl;
                   br.int_data= sqlite3_column_int(statement, clmn) ;

                   break;
                     case SQLITE_FLOAT:
					 data_type= SQLITE_FLOAT;
	 //std::cout << "SQLITE_FLOAT " << std::endl;
                  br.double_data= sqlite3_column_double(statement, clmn) ;

		  break;



                   }
                   br.data_size = field_size;
                   v_sbr.push_back(br);

                }
             }

             if(v_sbr.size() > 0 )bt.push_back(v_sbr);

        }
		///printf("step : %d\n\n",count_step);
    }


    sqlite3_finalize(statement);


	return SQLITE_DONE;


}

int  database::get_blob(const std::string &req ,void *data,int &blob_size )
{


   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    int res = SQLITE_ERROR;
    sqlite3_stmt *statement;
//double * buffer = new double[100];



    res = sqlite3_prepare_v2 (handle, req.c_str() , -1, &statement, NULL);

    if (res == SQLITE_OK)
    {
		int count_step = 0 ;
        while (sqlite3_step(statement) == SQLITE_ROW)
        {
            blob_size = sqlite3_column_bytes(statement, 0);
	        data = malloc(blob_size);
            data =(char  * )sqlite3_column_blob(statement, 0) ;
           //  buffer =(double  * )sqlite3_column_blob(statement, 0) ;
		   count_step++;

        }
		printf("step : %d\n\n",count_step);
    }

    if(blob_size > 0)
    sqlite3_finalize(statement);
    else
	{free(data);}

	return SQLITE_DONE;
}

void database::addToDB()
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    const char  *str = "CREATE TABLE IF NOT EXISTS Images (image1 BLOB);";
    int res = SQLITE_ERROR;
char * buffer = new char[100];

    res = sqlite3_exec(handle, str , NULL, NULL, NULL);

    sqlite3_stmt *updStmt =NULL;

    const char *sql = "INSERT INTO Images (image1) VALUES (?);";
    res = sqlite3_prepare_v2(handle, sql, -1, &updStmt, NULL);

    if(res!= SQLITE_OK)
    {
	 DBG_ERROR("sqlite3_prepare_v2  error\n");
    }


    memset(buffer,'A',10);
    res = sqlite3_bind_blob(updStmt, 1, buffer, 100*sizeof(double) , SQLITE_TRANSIENT);

    if((res = sqlite3_step(updStmt)) != SQLITE_DONE)
    {
		///errror
        sqlite3_reset(updStmt);
    }

    res = sqlite3_reset(updStmt);
    res = sqlite3_close(handle);
}

void database::readFromDB()
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
   const char *query = "SELECT image1 from Images";
    int res = SQLITE_ERROR;
    int len = 0;
 char * buffer = new char[100];

//double * buffer = new double[100];


    sqlite3_stmt *statement;
    res = sqlite3_prepare_v2 (handle, query , -1, &statement, NULL);

    if (res == SQLITE_OK)
    {
        if (sqlite3_step(statement) == SQLITE_ROW)
        {
            len = sqlite3_column_bytes(statement, 0);


	   if(len > 0)
             buffer =(char  * )sqlite3_column_blob(statement, 0) ;
           //  buffer =(double  * )sqlite3_column_blob(statement, 0) ;

          //std::cout << "len:"  << len << " ";
           printf("data : %c " ,(char) buffer[0]);





        }
    }
    if(buffer != 0)
    sqlite3_finalize(statement);

    res = sqlite3_close(handle);
}


//one call by row
static int callback(void *param, int argc, char **argv, char **azColName){
    //clear
    database * db=(database *) param;
    int i;
    m_row row;
    for(i=0; i<argc; i++)
    {
      row[azColName[i]]= argv[i] ? argv[i] : "NULL";
      //printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    db->tuples.push_back(row);
    //printf("***********************************\n");
    return 0;
}
v_tuples database::get_tuples()
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    return tuples;
}

int database::open ()
{
     #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    char * database_path=getenv("MBOX_DATABASE_PATH");
    if(database_path==NULL)
    {
        printf("[DB] MBOX_DATABASE_PATH=NULL \n");
        return this->open(DEFAULT_DATABASE_PATH);
     }
    else
    {
        printf("[DB] MBOX_DATABASE_PATH=%s\n",database_path);
        return this->open(database_path);

    }

}

int database::open (const std::string &file_name)
{

  #ifdef TRACE_DB_SQLITE3
   DBG_TRACE_P(file_name);
    #endif
 DBG_TRACE_P(file_name.c_str());
  int retval = sqlite3_open (file_name.c_str() ,&handle);
  if (retval)
    {
      DBG_ERROR("database conection error\n");
      return -1;
    }
  else
    {
      DBG_SQL("database open ok \n");
      return 1;
    }
}
int database::open (const char * file_name)
{

 #ifdef TRACE_DB_SQLITE3
   DBG_TRACE_P(file_name);
    #endif

  int retval = sqlite3_open (file_name ,&handle);
  if (retval)
    {
      DBG_ERROR("database conection error\n");
      return -1;
    }
  else
    {
      DBG_SQL("database open ok \n");
      return 1;
    }
}

int database::exec (const char *q)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
  tuples.clear();
  //printf("SIZE %d\n",tuples.size());
  int retval = sqlite3_exec (handle, q,callback , (void * )this, 0);
  //printf("SIZE %d\n",tuples.size());

  return retval;
}

int database::exec (const std::string &q)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
  tuples.clear();
  //printf("SIZE %d\n",tuples.size());
  int retval = sqlite3_exec (handle, q.c_str(),callback , (void * )this, 0);
  //printf("SIZE %d\n",tuples.size());

  return retval;
}
int database::open_exec(const std::string &q)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    int ret=open();
    if(ret == -1)
        return ret;
    ret = exec(q.c_str());
    //DBG_SQL("database close \n");
    sqlite3_close(handle);
    return ret;
}
int database::open_exec(const char *q)
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    int ret=open();
    if(ret == -1)
        return ret;
    ret = exec(q);
    //DBG_SQL("database close \n");
    sqlite3_close(handle);
    return ret;
}
void database::close()
{

   #ifdef TRACE_DB_SQLITE3
    DBG_TRACE;
    #endif
    sqlite3_close(handle);
    //DBG_SQL("database close \n");

}
