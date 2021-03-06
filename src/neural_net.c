/*
 * This file contains the definitions
 * of the procedures regarding the
 * neural network data structure.
 *
 * @author: Endri Kastrati
 * @date:   27/08/2017
 *
 */



/*
 * Including the standard utilities library,
 * the standard string manipulation library,
 * the standard assertions library and the
 * "neural_net.h" header file that contains
 * datatype definitions and function prototypings
 * of procedures regarding the neural network data
 * structure.
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "neural_net.h"






/*
 * @COMPLEXITY: O(l*m*n)    Where l is the number of layers
 *                          in the neural network and  ( m x n) are
 *                          the dimensions of the largest synaptic
 *                          weights matrix in the neural network.
 *
 * The function neural_net_create() takes one argument as
 * parameter,namely a neural_config_t data structure.It
 * instantiates a neural_net_t data structure by allocating
 * memory for it and it's components.Once memory has been
 * allocated the fields are instantiated and the arguments
 * are assigned to the corresponding components of the net.
 *
 * @param:  neural_config_t         *config
 * @return: neural_net_t            *
 *
 */

neural_net_t *neural_net_create(neural_config_t *config)
{
    // declaring a new neural network data
    // structure and checking whether the
    // given configuration is null or not.
    neural_net_t *new_nn=NULL;
    llint i; assert(config!=NULL);

    
    // Allocating memory for a new neural network data structure
    // and checking whether there was sufficient memory available.
    // Once the neural network has been instantiated we assign the
    // address of the given configuration structure to it's config
    // component.
    new_nn=(neural_net_t *)malloc(sizeof(*new_nn));
    assert(new_nn!=NULL); new_nn->config=config;

    
    // Based on the given configuration settings we allocate memory
    // for an array of neural layer data structures.
    new_nn->layers=(neural_layer_t **)malloc(config->nlayers*sizeof(neural_layer_t *));
    assert(new_nn->layers!=NULL);
    
    
    // Iterating over the newly created array
    // to instantiate each neural layer cell.
    for (i=0;i<config->nlayers;i++)
    {
        if (i==0)
        {
            // If we are at the first layer we need to specify as
            // synaptic weights the number of input signals and 
            // set the type of the layer to hidden.
            new_nn->layers[i]=neural_layer_create(config->neurons[i],
                config->signals,HIDDEN_NEURAL_LAYER);
            continue;
        }

        if (i+1==config->nlayers)
        {
            // If we are at the final layer ( output layer ) we need
            // to specify as synaptic weights the number of neurons
            // of the previous layer and set the type of the current
            // layer to output.
            new_nn->layers[i]=neural_layer_create(config->neurons[i],
                config->neurons[i-1]+1,OUTPUT_NEURAL_LAYER);
            continue;
        }
        
        // For the rest of the hidden layer we specify as synaptic
        // weights the number of neurons of the previous layer and 
        // set the type of the current layer to hidden.
        new_nn->layers[i]=neural_layer_create(config->neurons[i],
            config->neurons[i-1]+1,HIDDEN_NEURAL_LAYER);
    }
    
    // Once everything has been completed we return
    // the address of the newly created neural net.
    return new_nn;
}





/*
 * @COMPLEXITY: O(l*m*n)    Where l is the number of layers
 *                          in the neural network and the
 *                          ( m x n ) the dimensiosn of the
 *                          largest synaptic weights marix.
 * 
 * The static function forward_propagate() takes two immutable
 * void pointers as parameters and casts the first one into
 * a neural_net_t pointer and the second one into a gsl_vector 
 * pointer.Once the casting has been completed the given vector
 * is fetched as input signals into the neural network.We start
 * by forward propagating the corresponding inputs and outputs 
 * at each layer of the neural network until we reach the output
 * layer.Once the output layer has been reached the corresponding
 * output signals have been estimated.
 *
 * @param:  const void      *n
 * @param:  const void      *v
 * @return: void
 *
 */

static void forward_propagate(const void *n,const void *v)
{
    // Variable declarations and initializations,
    // type assertions and type castings.
    double wij,vi; double temp,value;
    size_t i,j,l,s; double sum=0.0;
    assert(n!=NULL && v!=NULL);
    neural_net_t *nn=NULL; gsl_vector *vv=NULL;
    nn=(neural_net_t *)n; vv=(gsl_vector *)v;
    
    // Beginning the forward propagation process
    // by iterating through each layer of the network.
    for (l=0;l<nn->config->nlayers;l++)
    {
        // Retrieving the synaptic weights matrix,the
        // linear aggregators matrix and the signals output
        // matrix for the current neural layer data structure.
        gsl_matrix *W=neural_layer_getW(nn->layers[l]);
        gsl_matrix *I=neural_layer_getI(nn->layers[l]);
        gsl_matrix *Y=neural_layer_getY(nn->layers[l]);

        // If we are not at the first neural layer
        // retrieve the signals output matrix from
        // the previous neural layer.
        gsl_matrix *prevY=NULL;
        if (l>0) { prevY=neural_layer_getY(nn->layers[l-1]); }
        
        // Iterating over the synaptic weights
        // matrix data structure and the corresponding
        // input signals matrix and calculating the
        // linear aggregators and output signals.
        for (i=0;i<W->size1;i++)
        {
            // Calculating the linear aggregator
            // for the current i neuron in the layer
            // using the following formula:
            //
            //  if we are at the first neural layer
            //
            //          Ii = Sum ( W(i,j) * vv(j) )
            //
            //  Otherwise,for the rest neural layers
            //
            //          Ii = Sum ( W(i,j) * prevY(j,0) )
            // 
            sum=0.0;
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
                if (l==0) { vi=gsl_vector_get(vv,j); }
                else { vi=gsl_matrix_get(prevY,j,0); }
                sum+=wij*vi;
            }
            
            // Calculating the linear aggregator value
            // for the current i neuron in the l layer.
            gsl_matrix_set(I,i,0,sum);
            temp=gsl_matrix_get(I,i,0);

            // Fetching the above value into the activation function
            // and the result is stored into the corresponding output
            // signals matrix cell.We also make sure that we are not
            // at the final layer of the network,since each signals
            // output matrix at the hidden layers has an extra cell
            // containing the bias factor.
            value=nn->config->activate(&temp,&nn->config->alpha,&nn->config->beta);
            if (nn->config->nlayers==l+1) { s=i; } else { s=i+1; }
            gsl_matrix_set(Y,s,0,value);

        }
        
        // If we are at the hidden layers insert the bias factor -1
        // at the beginning of the signals output matrix.
        if (nn->config->nlayers>l+1) { gsl_matrix_set(Y,0,0,-1.0); }
    } return;
}




/*
 * @COMPLEXITY: O(r*l*m*n)      Where r is the number of rows in the
 *                              input dataset,l is the number of layers
 *                              in the neural network and ( m x n ) are
 *                              the dimensions of the largest synaptic
 *                              weights matrix.
 * 
 * The function neural_net_predict() takes two arguments as parameters.
 * The first argument is an neural network data structure while the
 * second argument is a matrix that contains the input signals dataset.
 * This function fetches every row from the input signals matrix into
 * the neural network and retrieves the output and stores it into the
 * results matrix.The results matrix is allocated in the heap and thereby
 * the user has to make sure he deallocates it when it is not needed anymore.
 *
 * @param:  neural_net_t    *nn
 * @param:  gsl_matrix      *data
 *
 */

gsl_matrix *neural_net_predict(neural_net_t *nn,gsl_matrix *data)
{
    // Variable declarations
    // type assertions and
    // default instantiations.
    size_t i;
    assert(nn!=NULL && data!=NULL);
    gsl_matrix *results_matrix=NULL;
    gsl_vector_view dest_vector,src_vector;
    gsl_vector *dest=NULL,*src=NULL;
    gsl_vector_view row_vector; gsl_matrix *output_Y=NULL;

    // Retrieving the outputs signal matrix for the output layer
    // of the neural network and allocating memory for the new
    // results matrix based on the dimensions of the Y matrix.
    output_Y=neural_layer_getY(nn->layers[nn->config->nlayers-1]);
    results_matrix=gsl_matrix_alloc(data->size1,output_Y->size1);
    
    // Iterating over the rows given datasets
    for (i=0;i<data->size1;i++)
    {
        // Reading the data at the first row
        // as a matrix view and applying the
        // forward propagation procdure to
        // predict or classify the corresponding
        // output signals values.
        row_vector=gsl_matrix_row(data,i); 
        forward_propagate(nn,&row_vector);

        
        // Copying the results of the output signals
        // matrix into a vector view and then copying
        // those values into the first row of the results
        // matrix.
        src_vector=gsl_matrix_column(output_Y,0);
        dest_vector=gsl_matrix_row(results_matrix,i);
        dest=(gsl_vector *)&dest_vector;
        src=(gsl_vector *)&src_vector;
        gsl_vector_memcpy(dest,src);
    }

    // Returning the estimated output
    // signals in the form of a matrix
    // data structure.
    return results_matrix;
}





/*
 * @COMPLEXITY: O(f(n))     where f(n) is the time complexity
 *                          of the given training function.
 *
 * The function neural_net_train() takes two arguments 
 * as parameters.The first argument is a neural network
 * data structure and the second a gsl matrix that contains
 * training examples including the desired output for each
 * sample.This function invokes the given training function
 * on the given arguments.
 *
 * @param:  neural_net_t        *nn
 * @param:  gsl_matrix          *data
 * @return: void
 *
 */

void neural_net_train(neural_net_t *nn,gsl_matrix *data)
{
    assert(nn!=NULL && data!=NULL);
    assert(nn->config->train!=NULL);
    nn->config->train(nn,data);
    return;
}





/*
 * @COMPLEXITY: Theta(l)    where l is the total number of
 *                          layers in the neural network.
 * 
 * The static function config_dump() takes two arguments
 * as parameters.The first argument is a neural configuration
 * data structure and the second one a stream data structure.
 * This function saves the field components of the configuration
 * datatype as binary into the given stream address.
 *
 * @param:  neural_config_t     *config
 * @param:  FILE                *f
 * @return: void
 *
 */

static void config_dump(neural_config_t *config,FILE *f)
{
    assert(config!=NULL && f!=NULL);
    fwrite(&config->nlayers,sizeof(llint ),1,f);
    fwrite(config->neurons,sizeof(llint ),config->nlayers,f);
    fwrite(&config->signals,sizeof(llint ),1,f);
    fwrite(&config->epsilon,sizeof(double ),1,f);
    fwrite(&config->eta,sizeof(double ),1,f);
    fwrite(&config->alpha,sizeof(double ),1,f);
    fwrite(&config->beta,sizeof(double ),1,f);
    fwrite(&config->epochs,sizeof(llint ),1,f);
    fwrite(&config->atype,sizeof(int ),1,f);
    return;
}




/*
 * @COMPLEXITY: O(m*n)  Where ( m x n ) are the dimensions
 *                      of the largest synpatic weights
 *                      matrix in the neural network.
 * 
 * The static function weights_dump() takes two arguments
 * as parameters.The first argument is a neural_net_t data
 * structure and the second argument is a stream data structure.
 * This function saves the synaptic weights matrices of the
 * neural network as binary into the given stream address.
 *
 * @param:  neural_net_t        *nn
 * @param:  FILE                *f
 * @return: void
 *
 */

static void weights_dump(neural_net_t *nn,FILE *f)
{
    size_t l;
    gsl_matrix *tempW=NULL;
    assert(nn!=NULL && f!=NULL);
    for (l=0;l<nn->config->nlayers;l++)
    {
        tempW=neural_layer_getW(nn->layers[l]);
        gsl_matrix_fwrite(f,tempW);
    } return;
}




/*
 * @COMPLEXITY: O(m*n)  Where ( m x n ) are the dimensions
 *                      of the largest synaptic weights
 *                      matrix in the neural network.
 * 
 * The function neural_net_dump() takes two arguments
 * as parameters,namely a neural_net_t data structure
 * and a directory name and dumps the configuration data
 * as well as the synaptic weights of the neural network
 * into the given directory folder.
 *
 * @param:  neural_net_t        *nn
 * @param:  char                *directory
 * @return: void
 *
 */

void neural_net_dump(neural_net_t *nn,char *directory)
{
    int len1,len2;
    FILE *f1=NULL,*f2=NULL;
    char *config_name="/config.bin";
    char *weights_name="/weights.bin";
    char *filepath1,*filepath2;
    assert(nn!=NULL && directory!=NULL);
    len1=strlen(directory);
    len2=strlen(config_name);
    filepath1=(char *)malloc((len1+len2+1)*sizeof(char ));
    strcpy(filepath1,directory);
    strcat(filepath1,config_name);
    len2=strlen(weights_name);
    filepath2=(char *)malloc((len1+len2+1)*sizeof(char ));
    strcpy(filepath2,directory);
    strcat(filepath2,weights_name);
    f1=fopen(filepath1,"wb");
    config_dump(nn->config,f1);
    fclose(f1); f1=NULL;
    f2=fopen(filepath2,"wb");
    weights_dump(nn,f2);
    fclose(f2); f2=NULL;
    free(filepath2);
    free(filepath1);
    return;
}
    


/*
 * @COMPLEXITY: O(l)    Where l is the total number
 *                      of neural layers in the neural
 *                      network data structure.
 * 
 * The static function config_load() takes two arguments
 * as parameters,namely a neural configuration data structure
 * and a stream data structure and loads the configuration data
 * stored in the corresponding binary file into the given config
 * data structure.
 *
 * @param:  neural_config_nt    *config
 * @param:  FILE                *f
 * @return: void
 *
 */

static void config_load(neural_config_t *config,FILE *f)
{
    size_t bytes=0; bytes+=0;
    assert(config!=NULL && f!=NULL);
    bytes=fread(&config->nlayers,sizeof(llint ),1,f);
    config->neurons=(llint *)malloc(config->nlayers*sizeof(llint ));
    assert(config->neurons!=NULL);
    bytes=fread(config->neurons,sizeof(llint ),config->nlayers,f);
    bytes=fread(&config->signals,sizeof(llint ),1,f);
    bytes=fread(&config->epsilon,sizeof(double ),1,f);
    bytes=fread(&config->eta,sizeof(double ),1,f);
    bytes=fread(&config->alpha,sizeof(double ),1,f);
    bytes=fread(&config->beta,sizeof(double ),1,f);
    bytes=fread(&config->epochs,sizeof(llint ),1,f);
    bytes=fread(&config->atype,sizeof(int ),1,f);
    return;
}



/*
 * @COMPLEXITY: O(m*n)  Where ( m x n ) are the dimensions
 *                      of the largest synaptic weights matrix
 *                      of the neural network data structure.
 * 
 * The static function weights_load() takes two arguments as
 * parameters,namely a neural network data structure and a
 * stream data structure and loads the synaptic weights matrices
 * stored in the corresponding binary file into the given neural
 * network data structure.
 *
 * @param:  neural_net_t    *nn
 * @param:  FILE            *f
 * @return: void
 *
 */

static void weights_load(neural_net_t *nn,FILE *f)
{
    size_t l;
    assert(nn!=NULL && f!=NULL);
    gsl_matrix *tempW=NULL;
    
    for (l=0;l<nn->config->nlayers;l++)
    {
        tempW=neural_layer_getW(nn->layers[l]);
        gsl_matrix_fread(f,tempW);
    } return;
}
        


/*
 * @COMPLEXITY: O(m*n)      Where ( m x n ) are the dimensions of
 *                          the largest synaptic weights matrix of
 *                          the neural network data structure.
 * 
 * The function neural_net_load() takes two arguments as parameters,
 * namely a neural configuration data structure and a directory name
 * and loads the binary data stored in the directory into the given
 * config datatype and then instantiates a new neural network data 
 * structure and loads the synaptic weights and the configuration data
 * into the newly instantiated neural network.Once the loading has been
 * completed the address of the newly created neural net is returned.
 *
 * @param:  neural_config_t     *config
 * @param:  char                *directory
 * @return: neural_net_t        *
 *
 */

neural_net_t *neural_net_load(neural_config_t *config,char *directory)
{
    int len1,len2;
    FILE *f1=NULL,*f2=NULL;
    assert(directory!=NULL && config!=NULL);
    char *config_name="/config.bin";
    char *weights_name="/weights.bin";
    char *filepath1,*filepath2;
    neural_net_t *new_nn=NULL;    
    len1=strlen(directory);
    len2=strlen(config_name);
    filepath1=(char *)malloc((len1+len2+1)*sizeof(char ));
    strcpy(filepath1,directory);
    strcat(filepath1,config_name);
    len2=strlen(weights_name);
    filepath2=(char *)malloc((len1+len2+1)*sizeof(char ));
    strcpy(filepath2,directory);
    strcat(filepath2,weights_name); 
    f1=fopen(filepath1,"rb");
    config_load(config,f1);
    fclose(f1); f1=NULL;
    new_nn=neural_net_create(config);
    f2=fopen(filepath2,"rb");
    weights_load(new_nn,f2);
    fclose(f2); f2=NULL;
    free(filepath1);
    free(filepath2);
    return new_nn;
}






/*
 * @COMPLEXITY: O(l)    where l is the number of layers.
 *
 * The function neural_net_free() takes only one argument
 * as parameter,namely a neural network data structure and
 * deallocates all memory associated with it and it's components.
 * The config component is not deallocated as it might have been
 * allocated in the stack or in the heap by the user in which
 * case he is obligated to deallocated it manually himself.
 *
 * @param:  neural_net_t    *nn
 * @return: void
 *
 */

void neural_net_free(neural_net_t *nn)
{
    llint i;
    assert(nn!=NULL);
    for (i=0;i<nn->config->nlayers;i++)
    {
        neural_layer_free(nn->layers[i]);
        nn->layers[i]=NULL;
    }

    free(nn->layers);
    nn->layers=NULL;
    free(nn); nn=NULL;
    return;
}

