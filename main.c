#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "neural_layer.h"


int print_matrix(FILE *f,const gsl_matrix *m);

int main(int argc,char *argv[])
{
    neural_layer_t *new_layer=NULL;
    new_layer=neural_layer_create(4,4,HIDDEN_NEURAL_LAYER);
    gsl_matrix *currentW=NULL;
    gsl_matrix *currentI=NULL;
    gsl_matrix *currentY=NULL;
    currentW=neural_layer_getW(new_layer);
    currentI=neural_layer_getI(new_layer);
    currentY=neural_layer_getY(new_layer);
    
    print_matrix(stdout,currentW);
    printf("Size of I: (%zd,%zd)\n",currentI->size1,currentI->size2);
    printf("Size of Y: (%zd,%zd)\n",currentY->size1,currentY->size2);
    neural_layer_free(new_layer);
    return 0;
}


int print_matrix(FILE *f, const gsl_matrix *m)
{
	int status, n = 0;

	for (size_t i = 0; i < m->size1; i++) {
		for (size_t j = 0; j < m->size2; j++) {
		    if ((status = fprintf(f, "%g ", gsl_matrix_get(m, i, j))) < 0)
		            return -1;
		    n += status;
		}

		if ((status = fprintf(f, "\n")) < 0)
		        return -1;
		n += status;
	}

	return n;
}

