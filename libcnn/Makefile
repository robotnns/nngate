TARGET = libcnn.so
#set true for compil with cuda
CUDACOMPIL = false

OBJS =database.o pixdb.o ts_log.o config_parser.o filter_init_db.o Vector.o nng_math.o nng_math_eig.o nng_pca.o Matrix4d.o $(SPECIFIC_OBJ)
INCLUDE_DIR = -I. -I../dlib-master

CC = g++ -g -std=c++0x $(INCLUDE_DIR)
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x  $(INCLUDE_DIR)
CFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x 
LDFLAGS = -lpthread -lsqlite3 -shared

ifeq ($(CUDACOMPIL),true)
 SPECIFIC_OBJ=Matrix2d_cu.o
 LDFLAGS +=  ../libkernel/libkernel.so 
 
else 
 SPECIFIC_OBJ=Matrix2d.o
endif


all: $(TARGET)

libcnn.so: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
