#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "neural_utils.h"
#include "neural_net.h"




static int print_matrix(FILE *f, const gsl_matrix *m)
{
	int status, n = 0;

	for (size_t i = 0; i < m->size1; i++) {
		for (size_t j = 0; j < m->size2; j++) {
		    if ((status = fprintf(f, "%.2g ", gsl_matrix_get(m, i, j))) < 0)
		            return -1;
		    n += status;
		}

		if ((status = fprintf(f, "\n")) < 0)
		        return -1;
		n += status;
	}

	return n;
}




int main(int argc,char *argv[])
{
    llint neurons[3];
    gsl_matrix *results=NULL;
    neural_net_t *nn=NULL;
    FILE *train_file=NULL;
    neural_config_t config;

    neurons[0]=2;
    neurons[1]=10;
    neurons[2]=1;
    config.neurons=neurons;
    config.nlayers=3;
    config.signals=2;
    config.epsilon=1e-08;
    config.eta=0.009;
    config.alpha=1.0;
    config.beta=0.0;
    config.epochs=200;
    config.activate=hyperbolic_function;
    config.derivative=hyperbolic_derivative;
    config.train=resilient_backpropagation;

    
    train_file=fopen("sin_pattern.data","r");
    gsl_matrix *data=gsl_matrix_alloc(122,3);
    gsl_matrix_fscanf(train_file,data);
    fclose(train_file);


    nn=neural_net_create(&config);
    neural_net_train(nn,data);

    gsl_matrix_view Xdata=gsl_matrix_submatrix(data,0,0,122,2);
    gsl_matrix *XX=NULL; XX=(gsl_matrix *)&Xdata;
    results=neural_net_predict(nn,XX);
    
    /*
    for (size_t i=0;i<results->size1;i++)
    {
        gsl_vector_view row=gsl_matrix_row(results,i);
        double max=gsl_vector_max(&row);
        for (size_t j=0;j<results->size2;j++)
        {
            double cell=gsl_vector_get(&row,j);
            cell=(cell>=max ? 1.0 : 0.0);
            gsl_vector_set(&row,j,cell);
        }
    }*/

    print_matrix(stdout,results);
    neural_net_free(nn);
    gsl_matrix_free(data);
    gsl_matrix_free(results);
    return 0;
}

