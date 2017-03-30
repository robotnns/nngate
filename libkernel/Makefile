TARGET = libkernel.so
OBJS = cudalib.o
INCLUDE_DIR = -I. -I../dlib-master

CUDACC=/usr/local/cuda-8.0/bin/nvcc 
CFLAGS += --ptxas-options=-v --compiler-options '-fPIC' --shared
LDFLAGS = --ptxas-options=-v --compiler-options '-fPIC' --shared
all: $(TARGET)

libkernel.so: $(OBJS)
	$(CUDACC) -o $@ $^ $(LDFLAGS)


%.o: %.cu
	$(CUDACC) $(CFLAGS) -c -o $@ $<

$(OBJS): Makefile

clean:
	rm -rf $(OBJS) $(TARGET)

distclean: clean
	rm -f *~

.PHONY: clean distclean
