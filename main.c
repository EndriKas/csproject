#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "neural_utils.h"
#include "neural_net.h"





int main(int argc,char *argv[])
{
    llint neurons[5];
    neural_net_t *nn=NULL;
    FILE *train_file=NULL;
    neural_config_t config;

    neurons[0]=3;
    neurons[1]=3;
    neurons[2]=3;
    neurons[3]=3;
    neurons[4]=1;
    config.neurons=neurons;
    config.nlayers=5;
    config.signals=4;
    config.epsilon=1e-08;
    config.eta=0.01;
    config.momentum=0.9;
    config.alpha=1;
    config.beta=0.0;
    config.epochs=1000;
    config.activate=linear_function;
    config.derivative=linear_derivative;
    config.train=backpropagation;

    
    train_file=fopen("sample.data","r");
    gsl_matrix *data=gsl_matrix_alloc(20,5);
    gsl_matrix_fscanf(train_file,data);
    fclose(train_file);
    nn=neural_net_create(&config);
    neural_net_train(nn,data);

    gsl_matrix_view Xdata=gsl_matrix_submatrix(data,0,0,20,4);
    neural_net_predict(nn,&Xdata);
    neural_net_free(nn);
    gsl_matrix_free(data);
    return 0;
}



