TARGET = cnn
OBJS = nn_main.o 
CC = g++ -g -std=c++0x 
INCLUDE_DIR = -I.  -I../libcnn/

CXXFLAGS =  -g -std=c++0x -D_REENTRANT $(INCLUDE_DIR)
#LDFLAGS = -lpthread /usr/lib/libmbox.so
LDFLAGS = -lpthread ../libcnn/libcnn.so
CXXFLAGS += -Wall -Werror 

all: $(TARGET)

cnn: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean

