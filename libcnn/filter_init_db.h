#ifndef FILTER_INIT_H
#define FILTER_INIT_H

#include "database.h"
#include <stdlib.h>
class filter_init_db
{
	public :
	double rand_a_b(double a, double b);
	void fill_random_tab(double * tab ,unsigned long size,double min , double max);
	void print_double_tab(double * tab ,unsigned long size);
	void  init_lvl(const std::string & lvl , int filter_count , int filter_size,double min,double max, database & db);
	//void  filter_init_db::print_filter(database & db,int lvl);
	void  print_filter(const std::string id_net,char * filter_id,char * lvl);
	void init_net_filter(const std::string id_net);
	inline void set_param_db_path(std::string db_path){_param_db_path = db_path;}
    private :
    std::string _param_db_path="../data/net_param.db";
};
#endif
