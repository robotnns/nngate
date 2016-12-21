#include "config_parser.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define LG_MAX 75

void config_str_split(const char * str,struct  elem * rtp_chan_list,int list_max_size, const char delim)
{
    int e_i=0;
    //init the len
    for(int i=0;i<list_max_size;i++)
    {
        memset(rtp_chan_list[i].str,0,MAX_STR_ELEM_SIZE);
        rtp_chan_list[i].len=0;
    }

    for (int i = 0 ; str[i]!=0;i++)
    {
       if (str[i]!=delim)
       {
            rtp_chan_list[e_i].str[rtp_chan_list[e_i].len]=str[i];
            rtp_chan_list[e_i].len++;
       }
       else 
       {
            e_i++;

       }
    }
}

const char *  config_get_value(const conf_t *  conf,const char * key )
{
    for(int i = 0 ; i< conf->nbr_rec ;i++)
    {
        if(strcmp(conf->c_rec[i].key,key)==0)
                    return conf->c_rec[i].value;

    }
    return NULL;
}

int  config_load(conf_t * conf,const char * file_name)
{
   char str[200];
   FILE *fp;
   int conf_index=0;

	 
   fp = fopen(file_name, "r");
   if(!fp) return 1; // bail out if file not found
   while(fgets(str,sizeof(str),fp) != NULL)
   {
      // strip trailing '\n' if it exists
      int len = strlen(str)-1;
     /* if(str[len] == '\n' ) 
        { str[len] = 0;continue;}*/
    //printf("\nSTR=> %s",str);
     if( (str[0]!='<'&& str[0]!='#')&&str[len] == '\n')
     {
	 str[len] = 0;
     //get the value name 
     int state=0;
     int key_index=0;
     int value_index=0;
     
     for(int i = 0 ; i < len+1;i++)
 	{	   
		if( str[i]==' ')
		{state=1;}
		else 
		{
			if(state)
			{
	//			printf("[%d]P=>'%c'",value_index,str[i]);
				conf->c_rec[conf_index].value[value_index]=str[i];
				value_index++;
			}
			else 
			{
	//			printf("[%d]ID=>'%c'",key_index,str[i]);
				conf->c_rec[conf_index].key[key_index]=str[i];
				key_index++;
			}	
		}
	} 
     conf_index++;
     
     //printf("LINE\n%s", str);
     }       
   }
   fclose(fp);
   conf->nbr_rec=conf_index+1;

   return 0;
}	


void str_to_bin(char * in,int len,int &out)
{
    out=0;
    int vm[255];
    memset(vm,0,255);
    vm['0']=0;vm['1']=1;vm['2']=2;vm['3']=3;
    vm['4']=4;vm['5']=5;vm['6']=6;vm['7']=7;
    vm['8']=8;vm['9']=9;
    vm['a']=10;vm['b']=11;vm['c']=12;vm['d']=13;
    vm['e']=14;vm['f']=15;
    int b_decal=0;
    for(int i=len-1; i!=-1 ;i--)
    {
          
         out|=(vm[(int)in[i]]<<b_decal);
    //     printf("in[%d]=> %c  vm=> %02X out=> %02X\n",i,in[i],vm[(int)in[i]], out); 
         b_decal+=4;
    }
    printf("in=> %s out=> %02X\n",in , out);
}
