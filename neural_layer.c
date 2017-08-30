/*
 * This file contains the definition
 * of functions regarding the neural
 * layer data structure.
 *
 * @author: Endri Kastrati
 * @date:   23/08/2017
 *
 */




/*
 * Including the standard utilities library,
 * the standard assertions library,the standard
 * time manipulation library,the gnu matrix library,
 * the gnu random number generation library and the
 * header file "neural_layer.h" that cotains datatype
 * definitions and function prototyping regarding the
 * neural layer data structure.
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include "neural_layer.h"




/*
 * @COMPLEXITY: O(m*n) where  ( m x n ) are the dimensions
 *              of the weights matrix for the current layer.
 * 
 * The function neural_layer_create() takes three arguments
 * as parameters.The first argument is the number of neurons
 * the layer has.The second argument is the total number of
 * synaptic weights that are connected to each neuron in the
 * layer.The third argument is the type of the neural layer,
 * which can be either a hidden layer or an output layer.This
 * function instantiates an neural_layer_t data structure by
 * allocating memory for it and it's components.It returns a
 * newly created neural layer data structure.
 *
 * @param:  llint               j
 * @param:  llint               i
 * @param:  int                 layer_type
 * @return: neural_layer_t      *
 *
 */

neural_layer_t *neural_layer_create(llint j,llint i,int layer_type)
{
    // Variable declarations and
    // default instantiations.
    size_t row,column,brow;
    time_t seed; double random;
    neural_layer_t *new_nl=NULL;
    gsl_rng *random_gen=NULL;


    // Allocating memory for a new instance of
    // the neural_layer_t data structure and
    // checking whether allocation failed or not.
    new_nl=(neural_layer_t *)malloc(sizeof(*new_nl));
    assert(new_nl!=NULL);


    // Creating a new gsl random number generator
    // and seeding it using the time function.
    random_gen=gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(random_gen,time(&seed));

    
    // allocating a new matrix data structure that
    // has j number of rows and i number of columns.
    // the value of j represents the total number of
    // neurons in the current layer, while the value
    // of i represents the total number of synpatic
    // weights connected to the jth neuron.
    new_nl->W=gsl_matrix_alloc(j,i);

    
    // allocating a new matrix data structure that
    // has j number of rows and one column.this matrix
    // contains the linear aggregators for each neuron
    // in the current layer.
    new_nl->I=gsl_matrix_calloc(j,1);

    
    // Depending on the type of the current layer
    // we allcate a new matrix data structure that
    // has brow number of rows and one column.This
    // matrix contains the output signals for each
    // neuron in the current layer.
    brow=(layer_type==HIDDEN_NEURAL_LAYER ? j+1 : j);
    new_nl->Y=gsl_matrix_calloc(brow,1);
    
    // Popullating the cells of the weights matrix
    // with uniform random numbers between (0,1).
    for (row=0;row<new_nl->W->size1;row++)
    {
        for (column=0;column<new_nl->W->size2;column++)
        {
            random=gsl_rng_uniform_pos(random_gen);
            gsl_matrix_set(new_nl->W,row,column,random);
        }
    }
    
    // Deallocating memory for the random number
    // generator and returning the newly created
    // neural layer data structure.
    gsl_rng_free(random_gen);
    random_gen=NULL;
    return new_nl;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function neural_layer_getW() takes one argument
 * as a parameter,namely a neural layer data structure
 * and returns the address of it's weights matrix field.
 * 
 * @param:  neural_layer_t      *nl
 * @return: gsl_matrix          *
 * 
 */

gsl_matrix *neural_layer_getW(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->W;
}


/*
 * @COMPLEXITY: Theta(1)
 *
 * The function neural_layer_getI() takes one argument
 * as parameter,namely a neural layer data structure
 * and returns the address of it's linear aggregators
 * matrix field.
 *
 * @param:  neural_layer_t      *nl
 * @return: gsl_matrix          *
 *
 */

gsl_matrix *neural_layer_getI(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->I;
}


/*
 * @COMPLEXITY: Theta(1)
 *
 * The function neural_layer_getY() takes one argument
 * as parameter,namely a neural layer data structure
 * and returns the address of it's output signals matrix
 * field.
 *
 * @param:  neural_layer_t      *nl
 * @return: gsl_matrix          *
 *
 */

gsl_matrix *neural_layer_getY(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->Y;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function neural_layer_free() takes one argument
 * as parameter,namely a neural layer data structure
 * and deallocates memory for it and it's components.
 *
 * @param:  neural_layer_t      *nl
 * @return: void
 *
 */

void neural_layer_free(neural_layer_t *nl)
{
    assert(nl!=NULL);
    gsl_matrix_free(nl->W);
    gsl_matrix_free(nl->I);
    gsl_matrix_free(nl->Y);
    free(nl); nl=NULL;
    return;
}

