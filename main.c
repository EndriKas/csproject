#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "neural_net.h"


int         print_matrix(FILE *f,const gsl_matrix *m);


void        backpropagation(const void *,const void *);
double      logistic_function(double x,double a,double b);
double      logistic_derivative(double x,double a,double b);
double      calculate_square_error(neural_net_t *nn,gsl_matrix *D,llint nsamples);



int main(int argc,char *argv[])
{
    llint neurons[3];
    neural_net_t *nn=NULL;
    FILE *train_file=NULL;
    neural_config_t config;

    neurons[0]=3;
    neurons[1]=3;
    neurons[2]=2;
    config.neurons=neurons;
    config.nlayers=3;
    config.signals=4;
    config.epsilon=1e-08;
    config.eta=0.001;
    config.momentum=0.9;
    config.alpha=0.0001;
    config.beta=0.9;
    config.epochs=0;
    config.train=backpropagation;

    
    train_file=fopen("sample.data","r");
    gsl_matrix *data=gsl_matrix_alloc(4,6);
    gsl_matrix_fscanf(train_file,data);
    fclose(train_file);
    nn=neural_net_create(&config);

    /*
    for (int i=0;i<nn->config->nlayers;i++)
    {
        gsl_matrix *wi=neural_layer_getW(nn->layers[i]);
        gsl_matrix *ii=neural_layer_getI(nn->layers[i]);
        gsl_matrix *yi=neural_layer_getY(nn->layers[i]);
        size_t wd1=wi->size1,wd2=wi->size2;
        size_t id1=ii->size1,id2=ii->size2;
        size_t yd1=yi->size1,yd2=yi->size2;
        printf("Layer %d: W=(%zd,%zd), I=(%zd,%zd), Y=(%zd,%zd)\n",
            i,wd1,wd2,id1,id2,yd1,yd2);
    }*/


    neural_net_train(nn,data);
    neural_net_free(nn);
    gsl_matrix_free(data);
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




void backpropagation(const void *n,const void *d)
{
    size_t k1,k2,n1,n2; double err_next,err_prev;
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_matrix *data=NULL; data=(gsl_matrix *)d;
    k1=0; k2=0; n1=data->size1; n2=nn->config->signals;
    gsl_matrix_view X=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    k1=0; k2=nn->config->signals; n1=data->size1;
    n2=data->size2-nn->config->signals;
    gsl_matrix_view Y=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    err_next=calculate_square_error(nn,&Y,data->size1);
    // TODO: Implement the back propagation algorithm
    return;
}



double calculate_square_error(neural_net_t *nn,gsl_matrix *D,llint nsamples)
{
    size_t i,j; double sum=0;
    double total_error=0; assert(nn!=NULL && D!=NULL);
    gsl_matrix *Y=neural_layer_getY(nn->layers[nn->config->nlayers-1]);

    for (i=0;i<D->size1;i++)
    {
        sum=0.0;
        for (j=0;j<D->size2;j++)
        {
            double di=gsl_matrix_get(D,i,j);
            double yi=gsl_matrix_get(Y,j,0);
            sum+=pow(di-yi,2);
        }

        total_error+=sum/2.0;
    }
    
    total_error=total_error/(double )nsamples;
    return total_error;
} 
