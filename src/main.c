#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "dataset.h"
#include "neural_utils.h"
#include "neural_net.h"




void        predictions_print(FILE *f,const gsl_matrix *m);
int         read_command_arguments(int argc,char **argv,dataset_t *ds,neural_config_t *config);
double      minmax_scaler(double min,double max,double x,double a,double b);
void        usage(void);






int main(int argc,char *argv[])
{
    neural_config_t     config;
    dataset_t           *dataset=NULL;
    neural_net_t        *neural_network=NULL;
    char                *filename=NULL;
    char                *dumpDir=NULL;
    FILE                *stream=NULL;
    
    
    filename=&argv[3][10];
    stream=fopen(filename,"r");

    dataset=dataset_create(stream,DATASET_CLASSIFY,minmax_scaler);
    fclose(stream);

    struct stat st={0};
    if (stat(&argv[4][11],&st)==-1) { mkdir(&argv[4][11],0700); }

    dumpDir=argv[4][11];

    config.signals=atoi(&argv[5][10]);

    config.nlayers=atoi(&argv[6][9]);

    config.neurons=(llint *)malloc(config.nlayers*sizeof(llint ));
    assert(config.neurons!=NULL);


    int step=0; char *substring=&argv[7][20];
    while (step<config.nlayers)
    {
        config.neurons[step++]=(llint )atoi(substring+1);
        substring=strstr(substring+1,",");
    }

    config.eta=0.5;
    config.epsilon=1e-08;
    config.epochs=1000;
    config.alpha=1.0;
    config.beta=0.0;
    config.train=backpropagation;
    config.activate=logistic_function;
    config.derivative=logistic_derivative;
    
    neural_network=neural_net_create(&config);
    neural_net_train(neural_network,dataset->data);
    
    gsl_matrix_view X=gsl_matrix_submatrix(dataset->data,0,0,
        dataset->data->size1,config.signals);

    gsl_matrix *results=neural_net_predict(neural_network,(gsl_matrix *)&X);
    predictions_print(stdout,results);

    gsl_matrix_free(results);
    neural_net_free(neural_network);
    dataset_free(dataset);
    free(config.neurons);
    return 0;
}







double minmax_scaler(double min,double max,double x,double a,double b)
{
    double result=a*((x-min)/(max-min))-b;
    return result;
}



void usage(void)
{
    static const char *const content=
        "usage:\n"
        "\n"
        "   For the training process of the neural network:\n"
        "       ./neuralnet --train ( --curve-fitting | --pattern-classification ) --in-file=<filepath> --dump-dir=<filepath> --signals=<number> \n"
        "           --layers=<number> --neurons-per-layer=<[ number, .. ]>  [--epsilon=<number>] [--eta=<number>] [--epochs=<number>] [--alpha=<number>] [--beta=<number>]\n"
        "\n"
        "   For the prediction process of the neural network:\n"
        "       ./neuralnet --predict --load-dir=<number> --in-file=<filepath> --out-file=<filepath> --dims=<[number,number]>\n"
        "\n"
        "Available options:\n"
        "   --train                             set execution type to training.\n"
        "   --predict                           set execution type to predicting.\n"
        "   --curve-fitting                     set training mode to curve fitting.\n"
        "   --pattern-classification            set training mode to pattern classification.\n"
        "   --in-file=<filepath>                the filepath from which to read the training data.\n"
        "   --dump-dir=<filepath>               the name of the directory to dump the trained network.\n"
        "   --load-dir=<filepath>               The name of the directory to load the trained network.\n"
        "   --out-file=<filepath>               The name of the file in which the predicted data will be stored.\n"
        "   --signals=<number>                  The number of input signals or features plus the bias factor.\n"
        "   --layers=<number>                   The number of layers for the neural network.\n"
        "   --neurons-per-layer=<[n1,n2,..]>    The number of neurons per layer.\n"
        "   [--epsion=<number>]                 The convernge rate ( optional ).\n"
        "   [--eta=<number>]                    The learning rate ( optional ).\n"
        "   [--epochs=<number>]                 The number of epochs for the training session ( optional ).\n"
        "   [--alpha=<number>]                  The first coefficient of the activation function ( optional ).\n"
        "   [--beta=<number>]                   The second coefficient of the activation function ( optional ).\n"
        "   --help                              Print the help message and quit program execution.\n"
        "\n"
        "author: (c), Endri Kastrati, email: endriau@gmail.com\n";
    fprintf(stderr,"%s",content);
    return;
}


void predictions_print(FILE *stream,const gsl_matrix *m)
{
    size_t i,j; double data;
    for (i=0;i<m->size1;i++)
    {
        for (j=0;j<m->size2;j++)
        {
            data=gsl_matrix_get(m,i,j);
            fprintf(stream,"%g ",data);
        } fprintf(stream,"\n");
    }
    return;
}
