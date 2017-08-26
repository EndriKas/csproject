

#ifndef NEURAL_NET_H
#define NEURAL_NET_H



#include "neural_layer.h"


typedef double      (*ActivationFn)(const double *);
typedef double      (*DerivativeFn)(const double *);
typedef void        (*TrainingFn)(const void *,const void *);


typedef struct
{
    llint               *neurons;
    llint               nlayers;
    llint               signals;
    double              epsilon;
    double              eta;
    double              momentum;
    double              alpha;
    double              beta;
    llint               epochs;
    ActivationFn        activate;
    DerivativeFn        derivative;
    TrainingFn          train;
} neural_config_t;



typedef struct
{
    neural_config_t     *config;
    neural_layer_t      **layers;
} neural_net_t;





neural_net_t        *neural_net_create(neural_config_t *config);
void                neural_net_train(neural_net_t *nn,gsl_matrix *data);
void                neural_net_free(neural_net_t *nn);


#endif
