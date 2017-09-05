/*
 * This file contains data type
 * definitions and function prototypings
 * for the neural network data structure.
 *
 * @author: Endri Kastrati
 * @date:   27/08/2017
 *
 */



/*
 * Using include guards to check if
 * the neural_net.h header file has
 * been included at least once.If it
 * hasn't the compiler copy-pastes
 * everything into the file that is
 * including it.If the file on the
 * other hand has been included the
 * compiler skips the contents entirely.
 *
 */

#ifndef NEURAL_NET_H
#define NEURAL_NET_H



/*
 * Including the neural_layer.h header file
 * that contains data type definitions and
 * function prototypings regarding the neural
 * layer data structure.
 *
 */

#include "neural_layer.h"



/*
 * Defining three new data types of function pointers
 * called ActivationFn,DerivativeFn and TrainingFn.
 * These function pointers provide an interface for
 * defining the activation function and it's derivative
 * as well as the training process for the neural net.
 * These functions have to written by the user and meet
 * the following criteria.
 *
 */

typedef double      (*ActivationFn)(const double *,const double *,const double *);
typedef double      (*DerivativeFn)(const double *,const double *,const double *);
typedef void        (*TrainingFn)(const void *,const void *);



/*
 * Defining a new data structure called neural_config_t
 * that represents the abstract concept of a neural net
 * configuration process where the number of layers and
 * neurons per layer is defined as well as constants about
 * the training process of the network such as the learning
 * rate,the convergence constant,momentum value,alpha and beta
 * coefficients for the activation function.It also contains
 * three function pointers that potentially will invoke the
 * activation function,it's derivative and the training function
 * which must be implemented by the user.
 *
 */

typedef struct
{

    llint               nlayers;                // The total number of hidden layers + output layer.
    llint               *neurons;               // An array indicating the number of neurons per layer.
    llint               signals;                // The total number of input signals.
    double              epsilon;                // The convergence constant.
    double              eta;                    // The learning rate for the training process.
    double              alpha;                  // The alpha coefficient for the activation/derivative function.
    double              beta;                   // The beta coefficient for the activation/derivative function.
    llint               epochs;                 // The number of epochs for the training process.
    ActivationFn        activate;               // A function pointer to the activation function.
    DerivativeFn        derivative;             // A function pointer to the derivative function.
    TrainingFn          train;                  // A function pointer to the training function.
} neural_config_t;



/*
 * Defining a new data structure called neural_net_t
 * that represents the abstract concept of an artificial
 * neural network that can learn tasks by considering
 * examples,generally without task-specific programming.
 * This advanced data structure has two fields as methods.
 * The first field is a neural_config_t data structure that
 * is described above.The second field is an array of 
 * neural_layer_t data structures that represents the
 * layers of the network and where each layer has its
 * corresponding number of neurons.
 *
 */

typedef struct
{
    neural_config_t     *config;
    neural_layer_t      **layers;
} neural_net_t;





/*
 * function prototyping of procedures regarding the
 * neural_net_t data structure such as create,train
 * free, etc...
 *
 */

neural_net_t        *neural_net_create(neural_config_t *config);
gsl_matrix          *neural_net_predict(neural_net_t *nn,gsl_matrix *signals);
void                neural_net_train(neural_net_t *nn,gsl_matrix *data);
void                neural_net_dump(neural_net_t *nn,char *directory);
neural_net_t        *neural_net_load(neural_config_t *config,char *directory);
void                neural_net_free(neural_net_t *nn);





/*
 * Once everything has been copy-pasted by
 * the compiler and the macro NEURAL_NET_H
 * has been defined the neural_net.h header
 * file will not be included more than once.
 */

#endif
