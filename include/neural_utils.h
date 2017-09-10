/*
 * This file contains function prototypings
 * regarding utility procedures for configuring,
 * training and deploying the neural network data
 * structure.
 *
 * @author: Endri Kastrati
 * @date:   27/08/2017
 *
 */




/*
 * Using include guards to check if
 * the neural_utils.h header file has
 * been included at least once.If it
 * hasn't the compiler copy-pastes
 * everything into the file that is
 * including it.If the file on the
 * other hand has been included the
 * compiler skips the contents entirely.
 *
 */

#ifndef NEURAL_UTILS_H
#define NEURAL_UTILS_H



/*
 * Defining macro constants representing
 * the different color values that can be
 * outputed from the standard output stream.
 *
 */

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"





/*
 * Funtion prototypings of utility procedures
 * that assist in the configuration process
 * training phase and deployment of the neural
 * network data structure.
 *
 */

void            backpropagation(const void *,const void *);
double          mean_square_error_calculate(const void *,const void *,const void *);
double          logistic_function(const void *,const void *,const void *);
double          logistic_derivative(const void *,const void *,const void *);
double          hyperbolic_function(const void *,const void *,const void *);
double          hyperbolic_derivative(const void *,const void *,const void *);
double          linear_function(const void *,const void *,const void *);
double          linear_derivative(const void *,const void *,const void *);




/*
 * Once everything has been copy-pasted by
 * the compiler and the macro NEURAL_UTILS_H
 * has been defined the neural_utils.h header
 * file will not be included more than once.
 *
 */

#endif

