#ifndef TS_LOG_H
#define TS_LOG_H
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#define FILE_COUNT 5
#define LOG_ACTIVE
#include <stdarg.h>
#include <stdio.h>
#include <mutex>
#define STRING_SIZE_DATE_FORMAT 30 
#define LOG(file,message,args...) ts_log::get_instance()->write_file(file,message,##args)

typedef enum{LOG_LVL_DEBUG=1,LOG_LVL_INFO=2,LOG_LVL_SQL=4,LOG_LVL_WARN=8,LOG_LVL_ERROR=16,LOG_LVL_FATAL=32,LOG_LVL_NET_INFO=64,LOG_LVL_NET_ERROR=128}LOG_LVL_TYPE;
#define LOG_LVL_INFO_STR "INFO"
#define LOG_LVL_WARN_STR "WARN"  
#define LOG_LVL_ERROR_STR "ERROR"
#define LOG_LVL_FATAL_STR "FATAL"
#define LOG_LVL_DEBUG_STR "DEBUG"
#define LOG_LVL_SQL_STR  "SQL"
#define LOG_LVL_NET_INFO_STR "NET_INFO"
#define LOG_LVL_NET_ERROR_STR "NET_ERROR"  


class ts_log
{
    public:
    static ts_log*  get_instance();
    ts_log();
    ~ts_log();
	int write_file_2();
    void init(unsigned char);
    inline void set_log_desactive(){is_log_active=false;};
    inline void set_log_active(){is_log_active=true;};
    inline void set_stdout_desactive(){is_stdout_active=false;};
    inline void set_stdout_active(){is_stdout_active=true;};
    inline void get_lvl(char *buffer,LOG_LVL_TYPE lvl_type);
    int open_file(char * logfilename);
    static ts_log *  _instance;
    int write_file(LOG_LVL_TYPE type,const char * log,...); 
    void set_lvl_to_log_str(char* str);
    inline void get_time(char*);
    //int write_file(FILE_TYPE type,const char * log,va_list parametres);
//    static std::mutex mutex_write;
    private:
    bool is_log_active;
    bool is_stdout_active;
    unsigned char log_lvl_flag;
    FILE * file_desc;


};

#endif // TS_LOG_H
