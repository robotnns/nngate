#include "ts_log.h"
#include <stdlib.h> 

static std::mutex mutex_write;

ts_log *ts_log::_instance = NULL;
ts_log*  ts_log::get_instance()
{
	if(ts_log::_instance== NULL)
	{
        mutex_write.lock();
        if(ts_log::_instance== NULL)
	    {

		ts_log::_instance = new   ts_log;
		_instance->init(0);
        }
        mutex_write.unlock();
	}
	return   ts_log::_instance;
}
void ts_log::set_lvl_to_log_str(char* str)
{
    if(strcmp(LOG_LVL_INFO_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_INFO;

    }
    else if(strcmp(LOG_LVL_DEBUG_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_DEBUG;

    }
    else if(strcmp(LOG_LVL_ERROR_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_ERROR;

    }
    else if(strcmp(LOG_LVL_SQL_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_SQL;

    }
    else if(strcmp(LOG_LVL_WARN_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_WARN;

    }
    else if(strcmp(LOG_LVL_FATAL_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_FATAL;

    }
	
	 else if(strcmp(LOG_LVL_NET_INFO_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_NET_INFO;

    }
	
	 else if(strcmp(LOG_LVL_NET_ERROR_STR,str)==0)
    {
        this->log_lvl_flag |= LOG_LVL_NET_ERROR;

    }

}
void ts_log::init(unsigned char log_flag)
{
    this->log_lvl_flag=log_flag;
    this->set_stdout_desactive();
    this->set_log_active();
}

int ts_log::open_file(char * logfilename)
{
    mutex_write.lock();
	file_desc=fopen(logfilename,"w");
    int ret=0;
    if(file_desc==NULL)
    ret = -1; 
    mutex_write.unlock();
    return ret;
}

#define LOG_MAX_TAILLE_MESSAGE 1000
int ts_log::write_file(LOG_LVL_TYPE lvl_type, const char * log,...)
{
    //printf("val=> %02x\n",log_lvl_flag);
     //si les logs sont desactivés on return
    if(!is_log_active)return 0 ;

    //si le file pointet vaut null
    if(this->file_desc==NULL)return 0;

    //et logic pour savoir si le flag est actif
    //si le flag est pas actif on ecrit pas dans le fichier
    if(!(lvl_type & this->log_lvl_flag))return 0; 

    mutex_write.lock(); 
    va_list parametres;
    va_start(parametres, log);
    char message[LOG_MAX_TAILLE_MESSAGE+1];
    memset(message,0,LOG_MAX_TAILLE_MESSAGE);
    vsnprintf(message, LOG_MAX_TAILLE_MESSAGE+1, log, parametres);
	int message_len=strlen(message);
	int size_to_write=STRING_SIZE_DATE_FORMAT + message_len;
	//bufer global lvl+date+message
	char glob_buffer[LOG_MAX_TAILLE_MESSAGE];
	memset(glob_buffer,0,size_to_write);
    //recupere le lvl 
    this->get_lvl(glob_buffer,lvl_type);
	//recupere la date courante
	this->get_time(glob_buffer);
	//concatene le buffer a ecrire
	strcat(glob_buffer,message);
	//recupere la taille a ecrire
	size_to_write=strlen(glob_buffer);
    va_end(parametres);	
    int nwrite=0;
    if(is_stdout_active)
    printf("%s",glob_buffer);
	nwrite=fwrite (glob_buffer , 1 , size_to_write , file_desc);
	if ( nwrite<1)
	{
		printf("FATAL LOG WRITE ERROR  \n");
	}
	//on ecrit le buffer de la lib C direct dans le fichier
	fflush( file_desc );
    mutex_write.unlock();
	return nwrite;
}

/*
int ts_log::write_file(LOG_LVL_TYPE lvl_type, const char * log,...)
{
G_MAX_TAILLE_MESSAGE
    
    //si les logs sont desactivés on return
    if(!log_active)return 0 ;

    //si le file pointet vaut null
    if(this->file_desc==NULL)return 0;

    //et logic pour savoir si le flag est actif
    //si le flag est pas actif on ecrit pas dans le fichier
    if(!(lvl_type & this->log_lvl_flag))return 0; 

    va_list parametres;
    va_start(parametres, log);
    char message[LOG_MAX_TAILLE_MESSAGE+1];
    memset(message,0,LOG_MAX_TAILLE_MESSAGE);
    vsnprintf(message, LOG_MAX_TAILLE_MESSAGE, log, parametres);
    //get the log lvl
    int message_len=strlen(message);
    char * log_lvl;
    switch (lvl_type)
    {


        case LOG_LVL_DEBUG: log_lvl =(char*)"[DEBUG]";break;
        case LOG_LVL_INFO: log_lvl =(char*)"[INFO]";break;
        case LOG_LVL_SQL: log_lvl =(char*)"[SQL]";break;
        case LOG_LVL_WARN: log_lvl =(char*)"[ARN]";break;
        case LOG_LVL_ERROR: log_lvl =(char*)"[ERROR]";break;
        case LOG_LVL_FATAL: log_lvl =(char*)"FATAL";break;
    }                         
	int size_to_write=STRING_SIZE_DATE_FORMAT+MAX_SIZE_STR_TYPE_LINE + message_len;
	//bufer global message+date
	char glob_buffer[size_to_write];
	memset(glob_buffer,0,size_to_write);
    //add the lvl log
   // strcat(glob_buffer,message);
	//recupere la date courante
	this->get_time(glob_buffer);
	//concatene le buffer a ecrire
	strcat(glob_buffer,message);
	//reval en of 0x0 string 
	size_to_write=strlen(glob_buffer);
    va_end(parametres);	
int nwrite=3;
	//int nwrite= fwrite (glob_buffer , 1 , size_to_write , file_desc );
	int nwrite= fwrite ("TOTO" , 1 , 4, file_desc );
	if ( nwrite<1)
	{
		printf("FATAL LOG WRITE ERROR  \n");
		return -1;
	}
	//on ecrit le buffer de la lib C direct dans le fichier
	fflush( file_desc );
	return nwrite;
}*/



// le buffer doit etre ala bonne taille pour la chaine

void ts_log::get_lvl(char *buffer,LOG_LVL_TYPE lvl_type)
{
    char * log_lvl;
     switch (lvl_type)
    {

        case LOG_LVL_DEBUG: log_lvl =(char*)"[DEBUG]";break;
        case LOG_LVL_INFO: log_lvl =(char*)"[INFO]";break;
        case LOG_LVL_SQL: log_lvl =(char*)"[SQL]";break;
        case LOG_LVL_WARN: log_lvl =(char*)"[WARN]";break;
        case LOG_LVL_ERROR: log_lvl =(char*)"[ERROR]";break;
        case LOG_LVL_FATAL: log_lvl =(char*)"[FATAL]";break;
		 case LOG_LVL_NET_INFO: log_lvl =(char*)"[NET_INFO]";break;
        case LOG_LVL_NET_ERROR: log_lvl =(char*)"[NET_ERROR]";break;
        default :log_lvl =(char*)"UNKNOW";break;
    } 
    strcat(buffer,log_lvl);

}
void ts_log::get_time (char *buffer)
{

    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    //or gmttime for utc
    timeinfo = localtime (&rawtime);

    char tmp_buf[50];
    memset(tmp_buf,0,50);
	sprintf(tmp_buf, "[%02d:%02d:%02d -- %02d/%02d/%d] ", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,timeinfo->tm_mday,timeinfo->tm_mday+1,timeinfo->tm_year+1900 );
    strcat(buffer,tmp_buf);
}
ts_log::ts_log()
{
    //ctor
}

ts_log::~ts_log()
{
    //dtor
}
