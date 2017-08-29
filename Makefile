SRC = main.c neural_net.c neural_utils.c neural_layer.c
OBJ = main.o neural_net.o neural_utils.o neural_layer.o
EXE = neuralnet



CC 		= gcc
CFLAGS	= -Wall -O2
CLIBS	= -lm -lgsl -lgslcblas



compile: $(OBJ) Makefile
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ) $(CLIBS)


clean:
	rm -f $(OBJ) $(EXE)



memcheck:
	valgrind --leak-check=full ./$(EXE)

