#ifndef H_DEBUG_
#define H_DEBUG_


#define COLOR(x) x
#define RED COLOR("\e[1m\e[31m")
#define GREEN COLOR("\e[1m\e[32m")
#define YELLOW COLOR("\e[1m\e[33m")
#define BLUE COLOR("\e[1m\e[34m")
#define NORMAL COLOR("\e[0m\e[39m")
#define CYAN COLOR("\e[1m\e[36m")

#define M_INFO BLUE "[INFO] " NORMAL
#define M_SQL  CYAN "[SQL] " NORMAL
#define M_WARN YELLOW "[WARNING] " NORMAL
#define M_ERROR RED "[ERROR] " NORMAL
#define M_FATAL RED "[FATAL] " NORMAL
#define M_TRACE RED "[TRACE] " NORMAL




#define DBG(x...) fprintf(stderr,x )
#define DBG_INFO(x...) fprintf(stderr,M_INFO x)
#define DBG_SQL(x...) fprintf(stderr,M_SQL  x)
#define DBG_WARN(x...) fprintf(stderr,M_WARN x)
#define DBG_ERROR(x...) fprintf(stderr,M_ERROR x)
#define DBG_FATAL(x...) fprintf(stderr,M_FATAL x)

#define DBG_TRACE printf("%s %s()  %s:%d \n",M_TRACE,__FUNCTION__,__FILE__, __LINE__)
#define DBG_TRACE_P(x...) printf("%s %s(%s)  %s:%d \n",M_TRACE,__FUNCTION__,x,__FILE__, __LINE__)
#endif
