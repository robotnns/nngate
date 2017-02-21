TARGET =main_test
OBJS = main_test.o
INCLUDE_DIR = -I. -I../../libcnn -I../../dlib-master/ -I../../libsfviewer/

CC = g++ -g -std=c++0x  
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x $(INCLUDE_DIR)
LDFLAGS =-lpthread ../../libcnn/libcnn.so ../../dlib-master/build/dlib/libdlib.so -lsfml-graphics -lsfml-window -lsfml-system 

all: $(TARGET)

main_test: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


%.o: %.c
	$(CC)  -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
