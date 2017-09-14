/*
 * This file contains the main execution
 * program and a set of helper functions
 * regarding the training process of models
 * using a multilayer-perceptron network
 * as well as the deployment of such models.
 *
 * @author: Endri Kastrati
 * @date:   13/09/2017
 *
 */




/*
 * Including the standard input-output library,
 * the standard utilities library,the standard
 * assertions library,the standard mathematics
 * library,the standard string manipulation
 * library,the standard unix file library,
 * the stadard unix types library,the unix
 * standard symbolic constants and types
 * library,the dataset.h header file that
 * contains datatype definitions and function
 * prototypings for the dataset data structure,
 * the header file neural_utils.h that contains
 * helper functions for the neural network type
 * and the header file neural_net.h that contains
 * datatype definitions and function prototypings
 * regarding the neural network data structure.
 *
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
#include "dataset.h"
#include "neural_utils.h"
#include "neural_net.h"




#define EXECUTION_TRAIN             84          // Execution type training.
#define EXECUTION_PREDICT           80          // Execution type predicting.
#define MODE_CLASSIFICATION         67          // Training mode classification.
#define MODE_CURVEFITTING           85          // Training mode curve fitting.
#define NORMALIZE_YES               89          // Normalization flag to true.
#define NORMALIZE_NO                78          // Normalization flag to false.
#define ACTIVATION_LGST             1           // Logistic activation function.
#define ACTIVATION_LNR              2           // Linear activation function.
#define ACTIVATION_HTAN             3           // Hyperbolic tangent activation function.




/*
 * Function prototypings regarding helper functions
 * that read the command line arguments,apply the
 * necessary parsing functions and return the corresponding
 * datatype value.
 *
 */

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
double      read_momentum(int argc,char **argv);
llint       read_epochs(int argc,char **argv);
double      read_alpha(int argc,char **argv);
double      read_beta(int argc,char **argv);



/*
 * Function prototypings of procedures that assist
 * in the I/O aspects of the neural network data
 * structure.
 *
 */

void        resubstitution_testing(neural_net_t *nn,dataset_t *ds,int mode,int norm);
void        predictions_print(FILE *f,gsl_matrix *m);
void        predictions_format(gsl_matrix *m,dataset_t *ds,size_t ycol,int mode,int norm);
double      minmax_scaler(double min,double max,double x,double a,double b);
double      minmax_descaler(double min,double max,double x,double a,double b);
void        usage(void);






int main(int argc,char *argv[])
{
    int                 ds_type;            // the dataset_t type flag.
    neural_config_t     config;             // The neural configuration data structure.
    neural_net_t        *ann=NULL;          // The neural network data structure.
    dataset_t           *dataset=NULL;      // The dataset data structure.
    char                *filename=NULL;     // The file name variable. 
    char                *dumpDir=NULL;      // The dumping directory name variable.
    char                *loadDir=NULL;      // The  loading directory name variable.
    FILE                *stream=NULL;       // The file streaming variable.
    gsl_matrix          *results=NULL;      // The results matrix.

    
    // Check the total number of arguments and if there
    // is a mismatch or the help command line argument
    // has been provided invoke the usage() function 
    // and exit program execution.
    if (argc==1) { usage(); exit(EXIT_FAILURE); }
    if (strcmp(argv[1],"--help")==0) { usage(); exit(EXIT_FAILURE); }
    

    // Read the execution type and assign the
    // returned value to the type variable.
    int type=read_execution_type(argc,argv);

    // Read the mode type and assign the returned
    // value to the mode variable.
    int mode=read_training_mode(argc,argv);

    // Based on the value of the mode variable
    // initialize the ds_type variable.
    if (mode==MODE_CLASSIFICATION) { ds_type=DATASET_CLASSIFY; }
    if (mode==MODE_CURVEFITTING)   { ds_type=DATASET_PREDICT;  }

    // Read the normalization flag value
    // and assign it to the norm variable.
    int norm=read_normalization(argc,argv);

    
    // There are two execution types that are supported by
    // the main program.The first one is the training type.
    // When we provide the "--train" flag in the command line
    // we are letting the program know that we want to train
    // a new neural network data structure.The second one is
    // the predicting type.When we provide the "--predict"
    // flag in the command line we are letting the program
    // know that we want to load an already trained model
    // for predicting purposes. 



    // Check if the value of the type variable has been
    // set to the value of the EXECUTION_TRAIN macro.
    if (type==EXECUTION_TRAIN)
    {
        // If so,read the name of the file that
        // contains the training dataset.If the
        // filename equals to "stdin",namely the
        // standard input stream,then we read from
        // the stdin stream,if not then we open
        // the given filename.
        filename=read_in_file(argc,argv);
        if (strcmp(filename,"stdin")==0)
        { 
            // If the user wants to fetch the training
            // dataset via unix piping then we assign
            // the stdin stream to the stream variable.
            // We instantiate a new dataset_t data structure
            // with the stream variable and the normalization 
            // and denormalization functions as parameters.
            stream=stdin;
            dataset=dataset_create(stream,ds_type,minmax_scaler,minmax_descaler);

            // If the normalization setting has been set to true
            // then we normalize the loaded dataset.
            if (norm==NORMALIZE_YES) { dataset_scale(dataset); }
        }
        else
        {
            // If the user wants fetch the training dataset
            // from a regular unix file stream then we open
            // the given filepath and assign the returned
            // stream data structure to the stream variable.
            // We instantiate a new dataset_t data structure
            // with the stream viarble and the normalization
            // and denormalization function as parameters.
            // Once the dataset has been loaded we deallocate
            // resources associated with the opened stream.
            stream=fopen(filename,"r");
            dataset=dataset_create(stream,ds_type,minmax_scaler,minmax_descaler);
            fclose(stream);

            // If the normalization setting has been set to
            // true then we normalize the loaded dataset.
            if (norm==NORMALIZE_YES) { dataset_scale(dataset); }
        }
        
        // Reading the name of the directory
        // where the trained neural network
        // will be saved as binary.
        dumpDir=read_dump_dir(argc,argv);

        // Reading the total number of signals ( features )
        // that will be fetched as input signals into the
        // neural network data structure.We increment this
        // field by one since we always at a bias value.
        config.signals=read_signals(argc,argv)+1;

        // Reading the total number of neural layers that
        // the neural network data structure will have.
        config.nlayers=read_nlayers(argc,argv);

        // Reading the total number of neurons that each
        // neural layer data structure will contain.
        config.neurons=read_neurons_per_layer(argc,argv,config.nlayers);

        // Reading the activation function type that will be
        // used during the training and predicting process of
        // the neural network data structure.
        config.atype=read_activation(argc,argv);

        // Reading the convergence rate ( loss ) that represents
        // the minimum difference value between two consecutive
        // mean square error values.When this value is reached
        // the neural network training procedure stops.
        config.epsilon=read_epsilon(argc,argv);

        // Reading the learning rate for the training process
        // of the neural network data structure.
        config.eta=read_eta(argc,argv);

        // Reading the momentum value for the optimization
        // process of the training procedure of the neural net.
        config.momentum=read_momentum(argc,argv);

        // Reading the total number of epochs for that represents
        // the limit for the training iterations of the neural net.
        config.epochs=read_epochs(argc,argv);

        // Reading the alpha coefficient for the activation function.
        config.alpha=read_alpha(argc,argv);

        // Reading the beta coefficient for the activation function.
        config.beta=read_beta(argc,argv);

        // Using the optimized version of the back-propagation
        // algorithm that uses the momentum parameter for faster
        // convergence.
        config.train=backpropagation; 
        

        // Based on the supplied activation function type
        // we assign the corresponding function pointer to
        // the activate field and derivative field of the
        // neural configuration data structure.
        if (config.atype==ACTIVATION_LGST)
        {
            // If request activation function is the logistic
            // function then we assign the address of the
            // logistic function and it's derivative to the
            // activate and derivate fields.
            config.activate=logistic_function;
            config.derivative=logistic_derivative;
        }
        else if (config.atype==ACTIVATION_LNR)
        {
            // If requested activation function is the linear
            // function then we assign the address of the
            // linear function and it's derivative to the
            // activate and derivative fields.
            config.activate=linear_function;
            config.derivative=linear_derivative;
        }
        else if (config.atype==ACTIVATION_HTAN)
        {
            // If requested activation function is the hyperbolic
            // tangent function then we assign the address of the
            // hyperbolic tangent function and it's derivative to
            // the activate and derivative fields.
            config.activate=hyperbolic_function;
            config.derivative=hyperbolic_derivative;
        } else {}

        
        // Creating a new instance of the neural network
        // data structure by passing the neural configuration
        // data structure as an argument.
        ann=neural_net_create(&config);


        // Once the neural network data structure has been
        // created and properly configured we begin the training
        // process using the read dataset.
        neural_net_train(ann,dataset->data);

        
        // Once the training process has been completed as well
        // we save the current instance of the neural network
        // data structure into the specified directory name.
        neural_net_dump(ann,dumpDir);

        // Applying A simple resubstitution test to check on
        // the performance of the trained neural network.
        resubstitution_testing(ann,dataset,mode,norm);

        // Checking whether the user has set the normalization
        // flag.If so we have to save in binary the mininum and
        // maximum values for each column in the training dataset.
        if (norm==NORMALIZE_YES) { dataset_dump_minmax(dataset,dumpDir); }


        // Deallocating all memory blocks associated with the
        // neural network data structure,the dataset data structure
        // and the neurons array associated with the configuration
        // data structure.
        neural_net_free(ann);
        dataset_free(dataset);
        free(config.neurons);
    }
    


    // Check if the value of the type variable is
    // equal to the value of the EXECUTION_PREDICT macro. 
    if (type==EXECUTION_PREDICT)
    {
        // If so,read the name of the file that contains
        // the unseen dataset values.If the filename equals
        // to "stdin" then we read from the standard input
        // file stream.
        filename=read_in_file(argc,argv);
        if (strcmp(filename,"stdin")==0)
        {
            // If filename equls to "stdin" there is no
            // need to open up a file stream,we read from
            // the standard input file stream.We create
            // a new instance of the dataset_t data structure
            // using the stream variable and the normalization
            // and denormalization functions as arguments.
            stream=stdin;
            dataset=dataset_create(stream,ds_type,minmax_scaler,minmax_descaler);
        }
        else
        {
            // If on the other hand the filename is a regular
            // unix file,we open up the file and assign the
            // returned stream data structure to the stream
            // variable.We create a new instance of the dataset_t
            // data structure using the stream variable and normalization
            // and denormalization functions as paarameters.Once the values
            // have been loaded we deallocated resources associated with the
            // opened stream data structure.
            stream=fopen(filename,"r");
            dataset=dataset_create(stream,ds_type,minmax_scaler,minmax_descaler);
            fclose(stream);
        }
        
        // Reading the name of the directory that contains
        // the save neural network data structure.
        loadDir=read_load_dir(argc,argv);
        

        // Checking if the normalization flag has been set.
        if (norm==NORMALIZE_YES)
        { 
            // If so we load the min max values from the 
            // specified directory and scale the values of
            // the dataset data structure.
            dataset_load_minmax(dataset,loadDir);
            dataset_scale(dataset);
        }
        

        // Loading the saved neural network data structure
        // from the specified directory name.
        ann=neural_net_load(&config,loadDir);

        // Based on the supplied activation function type
        // we assign the corresponding function pointer to
        // the activate field and derivative field of the
        // neural configuration data structure.
        if (config.atype==ACTIVATION_LGST)
        {
            // If request activation function is the logistic
            // function then we assign the address of the
            // logistic function and it's derivative to the
            // activate and derivate fields.
            config.activate=logistic_function;
            config.derivative=logistic_derivative;
        }
        else if (config.atype==ACTIVATION_LNR)
        {
            // If requested activation function is the linear
            // function then we assign the address of the
            // linear function and it's derivative to the
            // activate and derivative fields.
            config.activate=linear_function;
            config.derivative=linear_derivative;
        }
        else if (config.atype==ACTIVATION_HTAN)
        {
            // If requested activation function is the hyperbolic
            // tangent function then we assign the address of the
            // hyperbolic tangent function and it's derivative to
            // the activate and derivative fields.
            config.activate=hyperbolic_function;
            config.derivative=hyperbolic_derivative;
        } else {}
        

        // Fetching the given unseed data into the loaded
        // neural network data structure and storing the
        // corresponding output signals into the results
        // matrix data structure.
        results=neural_net_predict(ann,dataset->data);

        // Formating the output signals based on the given command
        // line parameters and printing them in a user-friendly format.
        predictions_format(results,dataset,config.signals,mode,norm);
        predictions_print(stdout,results);

        // Deallocating all memory blocks associated with
        // the results matrix,the neural network data structure,
        // the dataset data structure and the neurons array of
        // the neural configuration data structure.
        gsl_matrix_free(results);
        neural_net_free(ann);
        dataset_free(dataset);
        free(config.neurons);
    }

    // Return the value zero back to the operating system
    // indicating that everything went as expected and no
    // errors or problems were encountered during execution.
    return 0;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_execution_type() reads the
 * execution type from the command line argument and
 * parses its value into an intenger and returns it.
 * If there is an error the usage() function is invoked
 * and the program execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: int
 *
 */

int read_execution_type(int argc,char **argv)
{
    if (argc>=2 && strcmp(argv[1],"--train")==0)   { return EXECUTION_TRAIN;   }
    if (argc>=2 && strcmp(argv[1],"--predict")==0) { return EXECUTION_PREDICT; }
    usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_training_mode() reads the
 * training mode from the command line argument and
 * parses it's value into an integer and returns it.
 * If there is an error the usage() function is invoked
 * and the program execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: int
 *
 */

int read_training_mode(int argc,char **argv)
{
    if (argc>=3 && strcmp(argv[2],"--curve-fitting")==0)            { return MODE_CURVEFITTING;   }
    if (argc>=3 && strcmp(argv[2],"--pattern-classification")==0)   { return MODE_CLASSIFICATION; }
    usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_normalization() reads the
 * normalization flag from the command line argument
 * and parses it's value into an integer and returns it.
 * If there is an error the usage() function is invoked
 * and the program execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: int
 *
 */

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
        


/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_in_file() reads the in file
 * name from the command line argument and returns it.
 * If there is an error, the usage() function is invoked
 * and the program execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: char    *
 *
 */

char *read_in_file(int argc,char **argv)
{
    if (argc>=5 && strstr(argv[4],"--in-file=")!=NULL) { return &argv[4][10]; }
    usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_dump_dir() reads the name of the
 * dumping directory from the command line arguments and returns it.
 * If there is an error,the usage() function is invoked and the program
 * execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: char    *
 *
 */

char *read_dump_dir(int argc,char **argv)
{

    if (argc>=6 && strstr(argv[5],"--dump-dir=")!=NULL)
    {
        struct stat st={0};
        if (stat(&argv[5][11],&st)==-1) { mkdir(&argv[5][11],0700); }
        return &argv[5][11];
    } usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_load_dir() reads the loading directory
 * form the command line argument and returns it.If there is an
 * error ,the usage() function is invoked and the program exec
 * is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: char    *
 *
 */

char *read_load_dir(int argc,char **argv)
{
    if (argc>=6 && strstr(argv[5],"--load-dir=")!=NULL) { return &argv[5][11]; }
    usage(); exit(EXIT_FAILURE);
}
        


/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_signals() reads the total number of
 * input signals (features) from the command line argument,parses
 * it into an long long int and returns it.If there is an error,
 * the usage() function is invoked and program execution terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: llint
 *
 */

llint read_signals(int argc,char **argv)
{
    if (argc>=7 && strstr(argv[6],"--signals=")!=NULL) { return atoll(&argv[6][10]); }
    usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_nlayers() reads the number of
 * layers from the command line argument,parses it into
 * a long long int and returns it.If there is an error,
 * the usage() function is invoked and program execution
 * terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: llint
 *
 */

llint read_nlayers(int argc,char **argv)
{
    if (argc>=8 && strstr(argv[7],"--nlayers=")!=NULL) { return atoll(&argv[7][10]); }
    usage(); exit(EXIT_FAILURE);
}



/*
 * @COMPLEXITY: O(l)    Where l is the total number of layers.
 *
 * The helper function read_neurons_per_layer() reads a list
 * that contains the number of neurons for each layer l and
 * returns an array of long long ints with the values.If an
 * error occurs,the usage() function is invoked and the program
 * execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: llint   *
 *
 */

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



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_activation() reads the activation
 * function type,parses it as an integer and returns it.If an
 * error occurs,the usage() function is invoked and the program
 * execution is terminated.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: int
 *
 */

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


/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_epsilon() reads the convergence rate
 * value from the command line argument,parses it into a double
 * and returns it.If the "--epsilon" flag was not specified the
 * convergence value defaults to 1e-08.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: double
 *
 */

double read_epsilon(int argc,char **argv)
{
    if (argc>=11 && strstr(argv[10],"--epsilon=")!=NULL) { return atof(&argv[10][10]); }
    double epsilon=1e-08; return epsilon;
}




/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_eta() reads the learning rate value
 * from the command line argument,parses it into a double and
 * returns it.If the "--eta" flag was not specified the learning
 * rate value defaults to 0.1.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: double
 *
 */

double read_eta(int argc,char **argv)
{
    if (argc>=12 && strstr(argv[11],"--eta=")!=NULL) { return atof(&argv[11][6]); }
    double eta=0.1; return eta;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_momentum() reads the momentum value
 * from the command line argument,parses it into a double and
 * returns it.If the "--momentum" flag was not specified the
 * momentum value defaults to 0.01.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: double
 *
 */

double read_momentum(int argc,char **argv)
{
    if (argc>=13 && strstr(argv[12],"--momentum=")!=NULL) { return atof(&argv[12][12]); }
    double momentum=0.01; return momentum;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_epochs() reads the total number of epochs
 * from the command line arguent,parses it into a long long int and
 * returns it.If the "--epochs" flag was not specified the epochs
 * value defaults to 200.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: llint
 *
 */

llint read_epochs(int argc,char **argv)
{
    if (argc>=14 && strstr(argv[13],"--epochs=")!=NULL) { return atoll(&argv[13][9]); }
    llint epochs=200; return epochs;
}


/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_alpha() reads the alpha coefficient
 * for the activation function from the command line argument,
 * parses it into a double and returns it.If the "--alpha" flag
 * was not specified the alpha value defaults to 1.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: double
 *
 */

double read_alpha(int argc,char **argv)
{
    if (argc>=15 && strstr(argv[14],"--alpha=")!=NULL) { return atof(&argv[14][8]); }
    double alpha=1.0; return alpha;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The helper function read_beta() reads the beta coefficient
 * for the activation function from the command line argument,
 * parses it into a double and returns it.If the "--beta" flag
 * was not specified the beta value defaults to 0.
 *
 * @param:  int     argc
 * @param:  char    **argv
 * @return: double
 *
 */

double read_beta(int argc,char **argv)
{
    if (argc>=16 && strstr(argv[15],"--beta=")!=NULL) { return atof(&argv[15][7]); }
    double beta=0.0; return beta;
}




/*
 * @COMPLEXITY: Theta(1)
 *
 * The function minmax_scaler() takes five arguments as parameters.
 * The first two are the minimum and maximum values of some column j.
 * The third parameter is an element of the i of the column and a,b
 * are the values that specify the range of scaling e,g [0,1] or [-1,-1].
 * This function returns the new normalized value of x.
 *
 * @param:  double      min
 * @param:  double      max
 * @param:  double      x
 * @param:  double      a
 * @param:  double      b
 * @return: double
 *
 */
double minmax_scaler(double min,double max,double x,double a,double b)
{
    double result=a*((x-min)/(max-min))-b;
    return result;
}


/*
 * @COMPLEXITY: Theta(1)
 *
 * The function minimax_descaler() is the opposite function of the
 * minmax_scaler function.Instead of normalizing the given data it
 * denormalizes it back to it's original form.
 *
 * @param:  double      min
 * @param:  double      max
 * @param:  double      x
 * @param:  double      a
 * @param:  double      b
 * @return: double
 *
 */

double minmax_descaler(double min,double max,double x,double a,double b)
{
    double result=(((x+b)*(max-min))/a)+min;
    return result;
}




/*
 * @COMPLEXITY: O(m*n)      Where ( m x n ) are th dimensions
 *                          of the given matrix data structure.
 * 
 * The function predictions_print() takes three arguments as
 * parameters.The first argument is a file stream data structure
 * and the second argument is a gsl_matrix data structure.This
 * function prints the values of the matrix in a user-friendly
 * format.
 *
 * @param:  FILE            *stream
 * @param:  gsl_matrix      *m
 * @return: void
 *
 */

void predictions_print(FILE *stream,gsl_matrix *m)
{
    // variable declarations.
    size_t i,j; double data;

    // Iterating over the rows and
    // columns of the matrix and
    // printing it's values into the
    // specified stream.
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




/*
 * @COMPLEXITY: O(m*n)      Where ( m x n ) are the dimensions of the given matrix.
 *                          
 * 
 * The function predictions_format() takes five arguments as parameters.
 * The first argument is a gsl_matrix data structure that contains the
 * output signals of the neural network.The second argument is a dataset_t
 * data structure.The third argument is the starting index of the output
 * targets in the dataset.The last two are the execution type and training
 * mode of the program.This function formats the output signals of the given
 * predictions matrix based on the neural net configurations.
 *
 * @param:  gsl_matrix      *m
 * @param:  dataset_t       *ds
 * @param:  size_t          ycol
 * @param:  int             mode
 * @param:  int             norm
 * @return: void
 *
 */

void predictions_format(gsl_matrix *m,dataset_t *ds,size_t ycol,int mode,int norm)
{
    // Variable declarastions,type assertions.
    assert(m!=NULL && ds!=NULL);
    assert(mode==MODE_CLASSIFICATION || mode==MODE_CURVEFITTING);
    gsl_vector_view v; size_t i,j,index; double min,max,data;
    
    // Iterating over rows the given matrix data structure.
    for (i=0;i<m->size1;i++)
    {
        // Because the neural network program supports
        // two types of training,namely pattern classification
        // and curve fitting, we have to format the output signals
        // accordingly.
        

        // Checking if the training mode is classification.
        if (mode==MODE_CLASSIFICATION)
        {
            // If so,we want to find the cell in the
            // current row with the maximum value and
            // set that one to 1 and the rest of the cells
            // in the current row to zero.
            v=gsl_matrix_row(m,i);
            index=gsl_vector_max_index((gsl_vector *)&v); 
            for (j=0;j<m->size2;j++)
            {
                if (j==index) { gsl_vector_set((gsl_vector *)&v,j,1.0); }
                else          { gsl_vector_set((gsl_vector *)&v,j,0.0); }
            }
        }
        
        // Checking if the training mode is curve fitting.
        if (mode==MODE_CURVEFITTING && norm==NORMALIZE_YES)
        {
            // If so,we want to iterate over the elements
            // of the current row in the matrix and descale
            // them using the minimum and maximum values of
            // the dataset data structure and the descaler
            // function.
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



/*
 * @COMPLEXITY: O(m*n)      Where ( m x n ) are the dimensions of the
 *                          training set matrix data structure.
 */
void resubstitution_testing(neural_net_t *nn,dataset_t *ds,int mode,int norm)
{
    size_t i; int flag;
    double mse=0.0,rmse=0.0;
    double error=0.0,accuracy=0.0;
    llint correct=0,wrong=0;
    assert(nn!=NULL && ds!=NULL);
    gsl_matrix *results=NULL;
    gsl_matrix *data=ds->data;
    gsl_matrix_view X,Y;
    gsl_vector_view output_row;
    gsl_vector_view desired_row;
    X=gsl_matrix_submatrix(data,0,0,data->size1,nn->config->signals);
    Y=gsl_matrix_submatrix(data,0,nn->config->signals,data->size1,data->size2-nn->config->signals);
    results=neural_net_predict(nn,(gsl_matrix *)&X);
    predictions_format(results,ds,nn->config->signals,mode,norm);
 
    if (mode==MODE_CLASSIFICATION)
    {
        for (i=0;i<results->size1;i++)
        {
            output_row=gsl_matrix_row(results,i);
            desired_row=gsl_matrix_row((gsl_matrix *)&Y,i);
            flag=gsl_vector_equal((gsl_vector *)&output_row,(gsl_vector *)&desired_row);
            if (flag==1) { correct++; }
            else         { wrong++;   }
        }

        accuracy=(double )correct/(double )data->size1; error=(double )wrong/(double )data->size1;
        printf(WHT"TESTING VIA RESUBSTITUTION:"RESET" "GRN"ACCURACY"RESET" = %g, "RED"ERROR"RESET" = %g\n",accuracy,error);
    }
    else if (mode==MODE_CURVEFITTING)
    {
        mse=mean_square_error_calculate(nn,&Y,&data->size1); rmse=sqrt(mse);
        printf(WHT"TESTING VIA RESUBSTITUTION:"RESET" "RED" ROOT MEAN SQUARE ERROR"RESET" = %g\n",rmse);
    }

    gsl_matrix_free(results);
    return;
}




/*
 * @COMPLEXITY: Theta(1)
 *
 * The usage() function takes no arguments as parameters
 * and prints information on how to properly use this
 * program.
 *
 * @param:  void
 * @return: void
 *
 */

void usage(void)
{
    static const char *const content=
        "usage:\n"
        "\n"
        "   For the training process of the neural network:\n"
        "\n"
        "       ./neuralnet --train ( --curve-fitting | --pattern-classification ) --normalization=<yes|no> --in-file=<filepath> --dump-dir=<filepath> --signals=<number> --nlayers=<number>\n"
        "           --neurons-per-layer=<[ number, .. ]> --activation=<lnr|lgst|htan>  [--epsilon=<number>] [--eta=<number>] [--momentum=<number>] [--epochs=<number>] [--alpha=<number>] [--beta=<number>]\n"
        "\n"
        "   For the prediction process of the neural network:\n"
        "\n"
        "       ./neuralnet --predict ( --curve-fitting | --pattern-classification ) --normalization=<yes|no>  --in-file=<filepath> --load-dir=<filepath>\n"
        "\n"
        "Available options:\n"
        "   --train                             This flag sets the execution mode to training.\n"
        "   --predict                           This flag sets the execution mode to predicting.\n"
        "   --curve-fitting                     This flag sets the training process to curve fitting.\n"
        "   --pattern-classification            This flag sets the training process to pattern classification..\n"
        "   --normalization=<yes|no>            This flag sets the normalization of the given data to on/off.\n"
        "   --in-file=<filepath>                This flag sets the name of the file that contains the training dataset.\n"
        "   --dump-dir=<filepath>               This flag sets the name of the directory where the trained model will be stored.\n"
        "   --load-dir=<filepath>               This flag sets the name of the directory from which to load a trained model.\n"
        "   --signals=<number>                  This flag sets the number of input signals (features) the dataset contains.\n"
        "   --nlayers=<number>                  This flag sets the number of layers the neural network should have.\n"
        "   --neurons-per-layer=<[n1,n2,..]>    This flag sets the number of neurons per layer the neural network should have.\n"
        "   --activation=<lnr|lgst|htan>        This flag sets the type of activation function for training process.\n"
        "   [--epsilon=<number>]                This flag sets the mse convergence value for the training process.  ( optional ).\n"
        "   [--eta=<number>]                    This flag sets the learning rate for the training process.          ( optional ).\n"
        "   [--momentum=<number>]               This flag sets the momentum rate for the training process.          ( optional ).\n"
        "   [--epochs=<number>]                 This flag sets the number of epochs for the training process.       ( optional ).\n"
        "   [--alpha=<number>]                  This flag sets the first coefficient for the activation function.   ( optional ).\n"
        "   [--beta=<number>]                   This flag sets the second coefficient of the activation function.   ( optional ).\n"
        "   --help                              Print the help message and quit program execution.\n"
        "\n"
        "   **  The files containing the training dataset must have the total number of\n"
        "       rows and columns in the first line and second line respectively\n"
        "\n"
        "   **  The files containing the newly unseen dataset mut have the total number of\n"
        "       rows and columns in the first line and second line respectively and have no target column.\n"
        "\n"
        "author: (c), Endri Kastrati, email: endriau@gmail.com\n";
    fprintf(stderr,"%s",content);
    return;
}
