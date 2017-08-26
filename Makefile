compile:
	gcc -Wall  -O2 main.c neural_net.c neural_layer.c -o test -lgsl -lgslcblas


memcheck:
	valgrind --leak-check=full ./test

clean:
	rm test

