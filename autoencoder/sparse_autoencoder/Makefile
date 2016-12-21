TARGET =main_sparseAE
OBJS =SparseAutoencoder.o Softmax.o main_sparseAE.o
INCLUDE_DIR = -I. -I../../libcnn -I../../dlib-master/

CC = g++ -g -std=c++0x 
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x $(INCLUDE_DIR) 
LDFLAGS =-lpthread ../../libcnn/libcnn.so ../../dlib-master/build/dlib/libdlib.so

all: $(TARGET)

main_sparseAE: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


%.o: %.c
	$(CC)  -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
