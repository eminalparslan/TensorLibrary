CC = g++
CC_FLAGS = -Wall -Wextra -std=c++17 -g

main: main.cpp tensor.o nn.o tensor.h
	$(CC) $(CC_FLAGS) main.cpp tensor.o nn.o -o main

nn.o: nn.cpp nn.h
	$(CC) $(CC_FLAGS) nn.cpp -o nn.o -c

tensor.o: tensor.cpp tensor.h
	$(CC) $(CC_FLAGS) tensor.cpp -o tensor.o -c

