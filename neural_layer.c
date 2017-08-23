
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include "neural_layer.h"





neural_layer_t *neural_layer_create(llint j,llint i,int layer_type)
{
    size_t row,column,bcolumn;
    time_t seed; double random;
    neural_layer_t *new_nl=NULL;
    gsl_rng *random_gen=NULL;
    new_nl=(neural_layer_t *)malloc(sizeof(*new_nl));
    assert(new_nl!=NULL);
    random_gen=gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(random_gen,time(&seed));
    new_nl->W=gsl_matrix_alloc(j,i);
    new_nl->I=gsl_matrix_alloc(j,1);
    bcolumn=(type==HIDDEN_NEURAL_LAYER ? j+1 : j);
    new_nl->Y=gsl_matrix_alloc(bcolumn,1);
    
    for (row=0;row<new_nl->W->size1;row++)
    {
        for (column=0;column<new_nl->W->size2;column++)
        {
            random=gsl_rng_uniform_pos(random_gen);
            gsl_matrix_set(new_nl->W,row,column,random);
        }
    }
    
    gsl_rng_free(random_gen);
    random_gen=NULL;
    return new_nl;
}



gsl_matrix *neural_layer_getW(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->W;
}


gsl_matrix *neural_layer_getI(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->I;
}


gsl_matrix *neural_layer_getY(neural_layer_t *nl)
{
    assert(nl!=NULL);
    return nl->Y;
}


void neural_layer_free(neural_layer_t *nl)
{
    assert(nl!=NULL);
    gsl_matrix_free(nl->W);
    gsl_matrix_free(nl->I);
    gsl_matrix_free(nl->Y);
    free(nl); nl=NULL;
    return;
}

