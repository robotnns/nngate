TARGET =main_stacked_autoencoder
OBJS =../sparse_autoencoder/SparseAutoencoder.o ../sparse_autoencoder/SparseAutoencoderGrad.o ../sparse_autoencoder/Softmax.o ../sparse_autoencoder/SoftmaxGrad.o StackedAutoencoder.o StackedAutoencoderGrad.o main_stacked_autoencoder.o
INCLUDE_DIR = -I. -I../../libcnn -I../../dlib-master/ -I../sparse_autoencoder

CC = g++ -g -std=c++0x 
CXXFLAGS += -D_REENTRANT -fPIC -Wall -Werror -g -std=c++0x $(INCLUDE_DIR) 
LDFLAGS =-lpthread ../../libcnn/libcnn.so ../../dlib-master/build/dlib/libdlib.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


%.o: %.c
	$(CC)  -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
