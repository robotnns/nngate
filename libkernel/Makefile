TARGET = libkernel.so
OBJS =Vector.o Matrix2d.o
INCLUDE_DIR = -I. -I../dlib-master

CC = g++ -g -std=c++0x $(INCLUDE_DIR)
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x  $(INCLUDE_DIR)
CFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x 
LDFLAGS = 

all: $(TARGET)

libkernel.so: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
