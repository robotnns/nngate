TARGET = libcnn.so
OBJS =main.o pixel_matrix_viewer.o
INCLUDE_DIR = -I. -I../dlib-master

CC = g++ -g -std=c++0x $(INCLUDE_DIR)
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x  $(INCLUDE_DIR)
CFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x 
LDFLAGS = -lpthread -lsqlite3 -lsfml-graphics -lsfml-window -lsfml-system  -shared 

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
