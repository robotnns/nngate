#ifndef H_CONFIG_PARSER
#define H_CONFIG_PARSER
#define MW_CONF_REC 22
#define CONFIG_KEY_SIZE 64
#define CONFIG_VALUE_SIZE 64
typedef struct config_rec
{
    char key[CONFIG_KEY_SIZE];
    char value[CONFIG_VALUE_SIZE];

} conf_rec_t;

typedef struct conf
{
    int nbr_rec;
    conf_rec_t c_rec[MW_CONF_REC];

} conf_t;

#define MAX_STR_ELEM_SIZE 22
#define MAX_CHANNEL 32
struct elem{

        char str[MAX_STR_ELEM_SIZE];
        int len;
};
extern  void str_to_bin(char * in,int len,int &out);
extern void config_str_split(const char * str,struct  elem * rtp_chan_list,int max_list_size, const char delim);
extern int  config_load(conf_t * conf,const char * file_name);
extern const char *  config_get_value(const conf_t *  conf,const char * key );
#endif
