CC = nvcc
CC_RELEASE_FLAGS = --compiler-options -std=c++17 -O3 -I/usr/local/cuda/include -lcuda -lnvrtc
# -ffp-model=fast
CC_DEBUG_FLAGS = --compiler-options -Wall -std=c++17 -g -I/usr/local/cuda/include -lcuda -lnvrtc
# -fdiagnostics-show-template-tree
# -fsanitize=address 
CC_FLAGS = $(CC_DEBUG_FLAGS)

# -fopenmp
# clang -cl-std=CL2.0 -cl-single-precision-constant test.cl
# clang --target=amdgcn-amd-amdhsa -mcpu=gfx900 test.cl

main: main.cpp tensor.o nn.o kernels.o tensor.h nn.h optimizer.h
	$(CC) $(CC_FLAGS) main.cpp tensor.o nn.o kernels.o -o main
	
%.o: %.cpp %.h
	$(CC) $(CC_FLAGS) $< -o $@ -c
	
%.o: %.cu %.h
	$(CC) $(CC_FLAGS) $< -o $@ -dc
	
clean:
	rm -f *.o main
