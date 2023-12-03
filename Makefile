CC = g++
CC_RELEASE_FLAGS = -std=c++17 -O3 -ffp-model=fast
CC_DEBUG_FLAGS = -Wall -Wextra -pedantic -std=c++17 -glldb -fdiagnostics-show-template-tree
# -fsanitize=address 
CC_FLAGS = $(CC_DEBUG_FLAGS)

# -fopenmp
# clang -cl-std=CL2.0 -cl-single-precision-constant test.cl
# clang --target=amdgcn-amd-amdhsa -mcpu=gfx900 test.cl

main: main.cpp tensor.o nn.o tensor.h nn.h optimizer.h
	$(CC) $(CC_FLAGS) main.cpp tensor.o nn.o -o main

nn.o: nn.cpp nn.h
	$(CC) $(CC_FLAGS) nn.cpp -o nn.o -c

tensor.o: tensor.cpp tensor.h
	$(CC) $(CC_FLAGS) tensor.cpp -o tensor.o -c

