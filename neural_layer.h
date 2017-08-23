


#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H



#include <gsl/gsl_matrix.h>


#define HIDDEN_NEURAL_LAYER 72
#define OUTPUT_NEURAL_LAYER 79


typedef long long int   llint;

typedef struct
{
    gsl_matrix      *W;
    gsl_matrix      *I;
    gsl_matrix      *Y;
} neural_layer_t;






neural_layer_t      *neural_layer_create(llint j,llint i,int layer_type);
gsl_matrix          *neural_layer_getW(neural_layer_t *nl);
gsl_matrix          *neural_layer_getI(neural_layer_t *nl);
gsl_matrix          *neural_layer_getY(neural_layer_t *nl);
void                neural_layer_free(neural_layer_t *nl);




#endif

