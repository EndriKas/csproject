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
 * the standard assertions library and the
 * "neural_net.h" header file that contains
 * datatype definitions and function prototypings
 * of procedures regarding the neural network data
 * structure.
 *
 */

#include <stdlib.h>
#include <assert.h>
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



static void forward_propagate(const void *n,const void *v)
{
    double wij,vi; double temp,value;
    size_t i,j,l,s; double sum;
    assert(n!=NULL && v!=NULL);
    neural_net_t *nn=NULL; gsl_vector *vv=NULL;
    nn=(neural_net_t *)n; vv=(gsl_vector *)v;

    for (l=0;l<nn->config->nlayers;l++)
    {


        gsl_matrix *W=neural_layer_getW(nn->layers[l]);
        gsl_matrix *I=neural_layer_getI(nn->layers[l]);
        gsl_matrix *Y=neural_layer_getY(nn->layers[l]);
        gsl_matrix *prevY=NULL;
        if (l>0) { prevY=neural_layer_getY(nn->layers[l-1]); }

        for (i=0;i<W->size1;i++)
        {
            sum=0.0;
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
                if (l==0) { vi=gsl_vector_get(vv,j); }
                else { vi=gsl_matrix_get(prevY,j,0); }
                sum+=wij*vi;
            }

            gsl_matrix_set(I,i,0,sum);
            temp=gsl_matrix_get(I,i,0);
            value=nn->config->activate(&temp,&nn->config->alpha,&nn->config->beta);
            if (nn->config->nlayers==l+1) { s=i; } else { s=i+1; }
            gsl_matrix_set(Y,s,0,value);

        }
        
        if (nn->config->nlayers>l+1) { gsl_matrix_set(Y,0,0,-1.0); }
    } return;
}


static int print_matrix(FILE *f, const gsl_matrix *m)
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





gsl_matrix *neural_net_predict(neural_net_t *nn,gsl_matrix *data)
{
    size_t i;
    assert(nn!=NULL && data!=NULL);
    gsl_vector_view row_vector;

    for (i=0;i<data->size1;i++)
    {
        row_vector=gsl_matrix_row(data,i);
        forward_propagate(nn,&row_vector);
        gsl_matrix *output_Y=NULL;
        output_Y=neural_layer_getY(nn->layers[nn->config->nlayers-1]);
        print_matrix(stdout,output_Y);
    }
    return NULL;
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

