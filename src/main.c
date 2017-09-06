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


#define EXECUTION_TRAIN             84
#define EXECUTION_PREDICT           80
#define MODE_CLASSIFICATION         67
#define MODE_CURVEFITTING           85
#define NORMALIZE_YES               89
#define NORMALIZE_NO                78
#define ACTIVATION_LGST             1
#define ACTIVATION_LNR              2
#define ACTIVATION_HTAN             3



int         read_execution_type(int argc,char **argv);
int         read_training_mode(int argc,char **argv);
int         read_normalization(int argc,char **argv);
char        *read_in_file(int argc,char **argv);
char        *read_dump_dir(int argc,char **argv);
char        *read_load_dir(int argc,char **argv);
llint       read_signals(int argc,char **argv);
llint       read_nlayers(int argc,char **argv);
llint       *read_neurons_per_layer(int argc,char **argv,llint n);
int         read_activation(int argc,char **argv);
double      read_epsilon(int argc,char **argv);
double      read_eta(int argc,char **argv);
llint       read_epochs(int argc,char **argv);
double      read_alpha(int argc,char **argv);
double      read_beta(int argc,char **argv);



void        predictions_print(FILE *f,const gsl_matrix *m);
void        predictions_format(gsl_matrix *m,dataset_t *ds,size_t ycol,int mode,int norm);
double      minmax_scaler(double min,double max,double x,double a,double b);
double      minimax_descaler(double min,double max,double x,double a,double b);
void        usage(void);





int main(int argc,char *argv[])
{
    int                 ds_type;
    neural_config_t     config;
    neural_net_t        *ann=NULL;
    dataset_t           *dataset=NULL;
    char                *filename=NULL;
    char                *dumpDir=NULL;
    char                *loadDir=NULL;
    FILE                *stream=NULL;
    gsl_matrix          *results=NULL;

    
    if (argc==1) { usage(); exit(EXIT_FAILURE); }
    if (strcmp(argv[1],"--help")==0) { usage(); exit(EXIT_FAILURE); }
    
    int type=read_execution_type(argc,argv);
    int mode=read_training_mode(argc,argv);
    if (mode==MODE_CLASSIFICATION) { ds_type=DATASET_CLASSIFY; }
    if (mode==MODE_CURVEFITTING)   { ds_type=DATASET_PREDICT;  }
    int norm=read_normalization(argc,argv);


    if (type==EXECUTION_TRAIN)
    {
        filename=read_in_file(argc,argv);
        if (strcmp(filename,"stdin")==0)
        { 
            stream=stdin;
            dataset=dataset_create(stream,ds_type,minmax_scaler,minimax_descaler);
            if (norm==NORMALIZE_YES) { dataset_scale(dataset); }
        }
        else
        { 
            stream=fopen(filename,"r");
            dataset=dataset_create(stream,ds_type,minmax_scaler,minimax_descaler);
            if (norm==NORMALIZE_YES) { dataset_scale(dataset); }
            fclose(stream);
        }
        
        dumpDir=read_dump_dir(argc,argv);
        config.signals=read_signals(argc,argv)+1;
        config.nlayers=read_nlayers(argc,argv);
        config.neurons=read_neurons_per_layer(argc,argv,config.nlayers);
        config.atype=read_activation(argc,argv);
        config.epsilon=read_epsilon(argc,argv);
        config.eta=read_eta(argc,argv);
        config.epochs=read_epochs(argc,argv);
        config.alpha=read_alpha(argc,argv);
        config.beta=read_beta(argc,argv);
        config.train=backpropagation; 

        if (config.atype==ACTIVATION_LGST)
        {
            config.activate=logistic_function;
            config.derivative=logistic_derivative;
        }
        else if (config.atype==ACTIVATION_LNR)
        {
            config.activate=linear_function;
            config.derivative=linear_derivative;
        }
        else if (config.atype==ACTIVATION_HTAN)
        {
            config.activate=hyperbolic_function;
            config.derivative=hyperbolic_derivative;
        }


        ann=neural_net_create(&config); 
        neural_net_train(ann,dataset->data);
        neural_net_dump(ann,dumpDir);
        if (norm==NORMALIZE_YES) { dataset_dump_minmax(dataset,dumpDir); }
        neural_net_free(ann);
        dataset_free(dataset);
        free(config.neurons);
    }
    
    if (type==EXECUTION_PREDICT)
    {   
        filename=read_in_file(argc,argv);
        if (strcmp(filename,"stdin")==0)
        {
            stream=stdin;
            dataset=dataset_create(stream,ds_type,minmax_scaler,minimax_descaler);
        }
        else
        {
            stream=fopen(filename,"r");
            dataset=dataset_create(stream,ds_type,minmax_scaler,minimax_descaler);
            fclose(stream);
        }
         
        loadDir=read_load_dir(argc,argv);

        if (norm==NORMALIZE_YES)
        { 
            dataset_load_minmax(dataset,loadDir);
            dataset_scale(dataset);
        }

        ann=neural_net_load(&config,loadDir);


        if (config.atype==ACTIVATION_LGST)
        {
            config.activate=logistic_function;
            config.derivative=logistic_derivative;
        }
        else if (config.atype==ACTIVATION_LNR)
        {
            config.activate=linear_function;
            config.derivative=linear_derivative;
        }
        else if (config.atype==ACTIVATION_HTAN)
        {
            config.activate=hyperbolic_function;
            config.derivative=hyperbolic_derivative;
        }

        config.train=backpropagation;
        results=neural_net_predict(ann,dataset->data);
        predictions_format(results,dataset,config.signals,mode,norm);
        predictions_print(stdout,results);
        gsl_matrix_free(results);
        neural_net_free(ann);
        dataset_free(dataset);
        free(config.neurons);
    }   
    return 0;
}



int read_execution_type(int argc,char **argv)
{
    if (argc>=2 && strcmp(argv[1],"--train")==0)   { return EXECUTION_TRAIN;   }
    if (argc>=2 && strcmp(argv[1],"--predict")==0) { return EXECUTION_PREDICT; }
    usage(); exit(EXIT_FAILURE);
}


int read_training_mode(int argc,char **argv)
{
    if (argc>=3 && strcmp(argv[2],"--curve-fitting")==0)            { return MODE_CURVEFITTING;   }
    if (argc>=3 && strcmp(argv[2],"--pattern-classification")==0)   { return MODE_CLASSIFICATION; }
    usage(); exit(EXIT_FAILURE);
}



int read_normalization(int argc,char **argv)
{
    char *substring=NULL;
    if (argc>=4 && strstr(argv[3],"--normalization=")!=NULL)
    {
        substring=&argv[3][16];
        if (strcmp(substring,"yes")==0) { return NORMALIZE_YES; }
        if (strcmp(substring,"no")==0)  { return NORMALIZE_NO;  }
    } usage(); exit(EXIT_FAILURE);
}
        


char *read_in_file(int argc,char **argv)
{
    if (argc>=5 && strstr(argv[4],"--in-file=")!=NULL) { return &argv[4][10]; }
    usage(); exit(EXIT_FAILURE);
}


char *read_dump_dir(int argc,char **argv)
{

    if (argc>=6 && strstr(argv[5],"--dump-dir=")!=NULL)
    {
        struct stat st={0};
        if (stat(&argv[5][11],&st)==-1) { mkdir(&argv[5][11],0700); }
        return &argv[5][11];
    } usage(); exit(EXIT_FAILURE);
}


char *read_load_dir(int argc,char **argv)
{
    if (argc>=6 && strstr(argv[5],"--load-dir=")!=NULL) { return &argv[5][11]; }
    usage(); exit(EXIT_FAILURE);
}
        


llint read_signals(int argc,char **argv)
{
    if (argc>=7 && strstr(argv[6],"--signals=")!=NULL) { return atoll(&argv[6][10]); }
    usage(); exit(EXIT_FAILURE);
}


llint read_nlayers(int argc,char **argv)
{
    if (argc>=8 && strstr(argv[7],"--nlayers=")!=NULL) { return atoll(&argv[7][10]); }
    usage(); exit(EXIT_FAILURE);
}


llint *read_neurons_per_layer(int argc,char **argv,llint n)
{
    llint *neurons=NULL,step=0; char *substring=NULL;
    if (argc>=9 && strstr(argv[8],"--neurons-per-layer=")!=NULL)
    {
        neurons=(llint *)malloc(n*sizeof(llint ));
        assert(neurons!=NULL);
        substring=&argv[8][20];
        
        while (step<n)
        {
            neurons[step++]=(llint )atoll(substring+1);
            substring=strstr(substring+1,",");
        } return neurons;
    } usage(); exit(EXIT_FAILURE);
}



int read_activation(int argc,char **argv)
{
    if (argc>=10 && strstr(argv[9],"--activation=")!=NULL)
    {
        if (strcmp(&argv[9][13],"lgst")==0) { return ACTIVATION_LGST; }
        if (strcmp(&argv[9][13],"lnr")==0)  { return ACTIVATION_LNR;  }
        if (strcmp(&argv[9][13],"htan")==0) { return ACTIVATION_HTAN; }
        usage(); exit(EXIT_FAILURE);
    } usage(); exit(EXIT_FAILURE);
}



double read_epsilon(int argc,char **argv)
{
    if (argc>=11 && strstr(argv[10],"--epsilon=")!=NULL) { return atof(&argv[10][10]); }
    double epsilon=1e-08; return epsilon;
}


double read_eta(int argc,char **argv)
{
    if (argc>=12 && strstr(argv[11],"--eta=")!=NULL) { return atof(&argv[11][6]); }
    double eta=0.9; return eta;
}


llint read_epochs(int argc,char **argv)
{
    if (argc>=13 && strstr(argv[12],"--epochs=")!=NULL) { return atoll(&argv[12][9]); }
    llint epochs=1000; return epochs;
}

double read_alpha(int argc,char **argv)
{
    if (argc>=14 && strstr(argv[13],"--alpha=")!=NULL) { return atof(&argv[13][8]); }
    double alpha=1.0; return alpha;
}

double read_beta(int argc,char **argv)
{
    if (argc>=15 && strstr(argv[14],"--beta=")!=NULL) { return atof(&argv[14][7]); }
    double beta=0.0; return beta;
}





double minmax_scaler(double min,double max,double x,double a,double b)
{
    double result=a*((x-min)/(max-min))-b;
    return result;
}


double minimax_descaler(double min,double max,double x,double a,double b)
{
    double result=(((x+b)*(max-min))/a)+min;
    return result;
}



void usage(void)
{
    static const char *const content=
        "usage:\n"
        "\n"
        "   For the training process of the neural network:\n"
        "\n"
        "       ./neuralnet --train ( --curve-fitting | --pattern-classification ) --normalization=<yes|no> --in-file=<filepath> --dump-dir=<filepath> --signals=<number> \n"
        "           --nlayers=<number> --neurons-per-layer=<[ number, .. ]>  [--epsilon=<number>] [--eta=<number>] [--epochs=<number>] [--alpha=<number>] [--beta=<number>]\n"
        "\n"
        "   For the prediction process of the neural network:\n"
        "\n"
        "       ./neuralnet --predict ( --curve-fitting | --pattern-classification ) --normalization=<yes|no>  --in-file=<filepath> --load-dir=<filepath>\n"
        "\n"
        "Available options:\n"
        "   --train                             set execution type to training.\n"
        "   --predict                           set execution type to prediction.\n"
        "   --curve-fitting                     set training mode to curve fitting.\n"
        "   --pattern-classification            set training mode to pattern classification.\n"
        "   --normalization=<yes|no>            Whether to normalize the dataset or not.\n"
        "   --in-file=<filepath>                the filepath from which to read the training data.\n"
        "   --dump-dir=<filepath>               the name of the directory to dump the trained neural network.\n"
        "   --load-dir=<filepath>               The name of the directory from which to load the trained neural network.\n"
        "   --signals=<number>                  The number of input signals or features.\n"
        "   --nlayers=<number>                   The number of layers for the neural network.\n"
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



void predictions_format(gsl_matrix *m,dataset_t *ds,size_t ycol,int mode,int norm)
{
    assert(m!=NULL && ds!=NULL);
    assert(mode==MODE_CLASSIFICATION || mode==MODE_CURVEFITTING);
    gsl_vector_view v; size_t i,j,index; double min,max,data;
    
    for (i=0;i<m->size1;i++)
    {
        if (mode==MODE_CLASSIFICATION)
        {
            v=gsl_matrix_row(m,i);
            index=gsl_vector_max_index((gsl_vector *)&v);
            
            for (j=0;j<m->size2;j++)
            {
                if (j==index) { gsl_vector_set((gsl_vector *)&v,j,1.0); }
                else          { gsl_vector_set((gsl_vector *)&v,j,0.0); }
            }
        }
        
        if (mode==MODE_CURVEFITTING && norm==NORMALIZE_YES)
        {
            for (j=0;j<m->size2;j++)
            {
                min=gsl_vector_get(ds->minimums,ycol+j);
                max=gsl_vector_get(ds->maximums,ycol+j);
                data=gsl_matrix_get(m,i,j);
                data=ds->descaler(min,max,data,2.0,1.0);
                gsl_matrix_set(m,i,j,data);
            }
        }
    } return;
}
