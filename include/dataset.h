/*
 * This file contains datatype definitions
 * and function prototypings regarding the
 * dataset data stucture.
 *
 * @author: Endri Kastrati
 * @date:   11/09/2017
 *
 */






/*
 * Using include guards to check if
 * the dataset.h header file has
 * been included at least once.If it
 * hasn't the compiler copy-pastes
 * everything into the file that is
 * including it.If the file on the
 * other hand has been included the
 * compiler skips the contents entirely.
 *
 */

#ifndef DATASET_H
#define DATASET_H



/*
 * Defining two new macro constants
 * that represent the type of dataset.
 * A dataset can be either used for
 * classification purposes or regression.
 *
 */

#define DATASET_CLASSIFY    67
#define DATASET_PREDICT     80



/*
 * Including the matrix library from the
 * GNU scientific library that provides
 * data structure definitions for matrices
 * and an interface for matrix operations.
 * 
 */

#include <gsl/gsl_matrix.h>




/*
 * Defining a new function pointer called ScalerFn.This function
 * pointer provides an interface for creating scaling functions
 * that take as input four doubles.
 *
 */

typedef double  (*ScalerFn)(double ,double ,double ,double ,double );



/*
 * Defining a new data structure called dataset_t
 * that represents the abstract concept of a dataset
 * with rows and columns seperated by empty space and
 * that contains entirely numerical values.It has as
 * components two gsl_vectors that store the minimum
 * and maximum values of each column,the dimensions
 * of the dataset ( rows, column ),the matrix that
 * contains the dataset values and two function 
 * pointers to a scaling and descaling function.
 *
 */

typedef struct
{
    gsl_vector          *maximums;
    gsl_vector          *minimums;
    long long int       rows;
    long long int       columns;
    gsl_matrix          *data;
    int                 type;
    ScalerFn            scaler;
    ScalerFn            descaler;
} dataset_t;





/*
 * Functio prototypings of procedures regarding the
 * dataset data structure such as create,scale,free etc...
 *
 */

dataset_t           *dataset_create(FILE *f,int type,ScalerFn scaler,ScalerFn descaler);
void                dataset_dump_minmax(dataset_t *ds,char *directory);
void                dataset_load_minmax(dataset_t *ds,char *directory);
void                dataset_scale(dataset_t *ds);
void                dataset_free(dataset_t *ds);





/*
 * Once everything has been copy-pasted by the 
 * compiler and the macro DATASET_H has been
 * defined the dataset.h header file will not
 * be included more than once.
 *
 */

#endif
