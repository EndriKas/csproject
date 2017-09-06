#ifndef DATASET_H
#define DATASET_H


#define DATASET_CLASSIFY    67
#define DATASET_PREDICT     80



#include <gsl/gsl_matrix.h>


typedef double  (*ScalerFn)(double ,double ,double ,double ,double );


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




dataset_t           *dataset_create(FILE *f,int type,ScalerFn scaler,ScalerFn descaler);
void                dataset_dump_minmax(dataset_t *ds,char *directory);
void                dataset_load_minmax(dataset_t *ds,char *directory);
void                dataset_scale(dataset_t *ds);
void                dataset_free(dataset_t *ds);



#endif
