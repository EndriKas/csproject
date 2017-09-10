/*
 * This file contains data definitions
 * and function prototypings for the 
 * neural layer data structure.
 *
 * @author: Endri Kastrati
 * @date:   23/08/2018
 *
 */




/*
 * Using include guards to check if
 * the neural_layer.h header file has
 * been included at least once.If it
 * hasn't the compiler copy-pastes
 * everything into the file that is
 * including it.If the file on the
 * other hand has been included the
 * compiler skips the contents entirely.
 *
 */

#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H




/*
 * Including the matrix library from the
 * GNU scientific library that provides
 * data structure definitions for matrices
 * and an interface for matrix operations.
 * 
 */

#include <gsl/gsl_matrix.h>




/*
 * Defining two macro constants that
 * represent the different types of
 * neural layers,namely the hidden
 * layers and the output layer.
 *
 */

#define HIDDEN_NEURAL_LAYER 72
#define OUTPUT_NEURAL_LAYER 79




/*
 * Defining an alias for the native type long long int
 * called llint.Defining a new data structure called 
 * neural_layer_t that represents the abstract concept
 * of a layer of nodes in an artificial neural network.
 * The layer data type has three gsl matrices as fields.
 * The first field,namely W, is the weight's matrix for
 * the current layer.The second field,namely I, is the
 * matrix that contains the linear aggregators for each
 * neuron in the current layer.The third field,namley Y
 * is the matrix that contains the generated output signal
 * from each neuron in the layer.The fourth field,namely
 * D is the gradient value for each neuron in the layer.
 * 
 */

typedef long long int   llint;

typedef struct
{
    gsl_matrix      *W;     // The synaptic weights matrix
    gsl_matrix      *I;     // The linear aggregators matrix.
    gsl_matrix      *Y;     // The output signals matrix.
    gsl_matrix      *D;     // The gradient matrix.
    gsl_matrix      *O;     // The synaptic weights of the previous epoch.
} neural_layer_t;





/*
 * Function prototyping of procedures regarding the neural 
 * layer data structure, such as create,free  fields etc..
 *
 */

neural_layer_t      *neural_layer_create(llint j,llint i,int layer_type);
gsl_matrix          *neural_layer_getW(neural_layer_t *nl);
gsl_matrix          *neural_layer_getI(neural_layer_t *nl);
gsl_matrix          *neural_layer_getY(neural_layer_t *nl);
gsl_matrix          *neural_layer_getD(neural_layer_t *nl);
gsl_matrix          *neural_layer_getO(neural_layer_t *nl);
void                neural_layer_free(neural_layer_t *nl);





/*
 * Once everything has been copy-pasted by the 
 * compiler and the macro NEURAL_LAYER_H has been
 * defined the neural_layer.h header file will not
 * be included more than once.
 *
 */

#endif

