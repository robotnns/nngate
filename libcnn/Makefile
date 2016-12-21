TARGET = libcnn.so
OBJS =database.o pixdb.o ts_log.o config_parser.o filter_init_db.o cnn_util.o
INCLUDE_DIR = -I. -I../dlib-master

CC = g++ -g -std=c++0x $(INCLUDE_DIR)
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x  $(INCLUDE_DIR)
CFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x 
LDFLAGS = -lpthread -lsqlite3 -shared

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
