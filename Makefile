compile:
	gcc -Wall  -O2 main.c neural_net.c neural_layer.c neural_utils.c -o test -lgsl -lgslcblas -lm


memcheck:
	valgrind --leak-check=full ./test

clean:
	rm test

