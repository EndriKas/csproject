/*
 * This file contains the main execution
 * function for the multi-layer perceptron
 * network as well as helper functions.
 *
 * @author: Endri Kastrati
 * @date:   30/08/2017
 *
 */



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "neural_utils.h"
#include "neural_net.h"





void    predictions_print(FILE *,const gsl_matrix *);
int     read_command_arguments(int argc,char *argv[]);
void    usage(void);



int main(int argc,char *argv[])
{
    FILE *infile=NULL;
    char *filename=NULL;
    char *substring=NULL;
    size_t rows,columns,step;
    neural_config_t config;
    neural_net_t *neural_network=NULL;

    if (argc==2 && strcmp(argv[1],"--help")==0) { usage(); exit(EXIT_FAILURE); }
    if (argc>2 && strcmp(argv[1],"--train")==0)
    {
        if (strcmp(argv[2],"--curve-fitting")==0)
        {
            config.activate=hyperbolic_function;
            config.derivative=hyperbolic_derivative;
        }
        else if (strcmp(argv[2],"--pattern-classification")==0)
        {
            config.activate=logistic_function;
            config.derivative=logistic_derivative;
        } else { usage(); exit(EXIT_FAILURE); }

    
        if (strstr(argv[3],"--in-file=")!=NULL) { filename=&argv[3][10]; } 
        else { usage(); exit(EXIT_FAILURE); }
       

        if (strstr(argv[4],"--dump-dir=")!=0)
        {
            struct stat st = {0};
            if (stat(&argv[4][11],&st)==-1) { mkdir(&argv[4][11],0700); }
        } else { usage(); exit(EXIT_FAILURE); }


        if (strstr(argv[5],"--dims=")!=NULL)
        {
            rows=atoi(&argv[5][8]);
            substring=strstr(&argv[5][8],",");
            columns=atoi(substring+1);
        } else { usage(); exit(EXIT_FAILURE); }

        if (strstr(argv[6],"--signals=")!=NULL) { config.signals=atoi(&argv[6][10]); }
        else { usage(); exit(EXIT_FAILURE); }

        if (strstr(argv[7],"--layers=")!=NULL)
        { 
            config.nlayers=atoi(&argv[7][9]); 
            config.neurons=(llint *)malloc(config.nlayers*sizeof(llint ));
        }
        else { usage(); exit(EXIT_FAILURE); }
    

        if (strstr(argv[8],"--neurons-per-layer=")!=NULL)
        {
            step=0;
            substring=&argv[8][20];
            while (step<config.nlayers)
            {
                config.neurons[step++]=(llint )atoi(substring+1);
                substring=strstr(substring+1,",");
            }
        } else { usage(); exit(EXIT_FAILURE); }

        printf("input filename      = %s\n",filename);
        printf("dump directory      = %ss\n",&argv[4][11]);
        printf("dimensions          = (%lu,%lu)\n",rows,columns);
        printf("signals             = %lld\n",config.signals);
        printf("layers              = %lld\n",config.nlayers);
        printf("neurons per layer   = "); 
        for (step=0;step<config.nlayers;step++) { printf("%lld ",config.neurons[step]); }; printf("\n");
        
        config.epsilon=1e-08;
        config.eta=0.5;
        config.alpha=1.0;
        config.beta=0.0;
        config.epochs=200;
        config.train=backpropagation;
        infile=fopen(filename,"r");
        gsl_matrix *data=gsl_matrix_alloc(rows,columns);
        gsl_matrix_fscanf(infile,data);
        neural_network=neural_net_create(&config);
        neural_net_train(neural_network,data);

        gsl_matrix_view X=gsl_matrix_submatrix(data,0,0,data->size1,config.signals);
        gsl_matrix *results=neural_net_predict(neural_network,&X);
        predictions_print(stdout,results);
        free(config.neurons);
        fclose(infile);
        gsl_matrix_free(data);
        neural_net_free(neural_network);
    }

    return 0;
}




void usage(void)
{
    static const char *const content=
        "usage:\n"
        "\n"
        "   For the training process of the neural network:\n"
        "       ./neuralnet --train ( --curve-fitting | --pattern-classification ) --in-file=<filepath> --dump-dir=<filepath>  --dims=<[number,number]> --signals=<number> \n"
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
        "   --dump-dir=<filepath>               the name of the directory from which to either the trained network.\n"
        "   --load-dir=<filepath>               The name of the directory from which to load the trained network.\n"
        "   --out-file=<filepath>               The name of the file in which the predicted data will be stored.\n"
        "   --dims=<[number,number]>            The dimensions of the training dataset.\n"
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
