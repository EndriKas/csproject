#include <stdlib.h>
#include <assert.h>
#include "neural_net.h"






neural_net_t *neural_net_create(neural_config_t *config)
{
    llint i;
    assert(config!=NULL);
    neural_net_t *new_nn=NULL;
    new_nn=(neural_net_t *)malloc(sizeof(*new_nn));
    assert(new_nn!=NULL); new_nn->config=config;
    new_nn->layers=(neural_layer_t **)malloc(config->nlayers*sizeof(neural_layer_t *));
    assert(new_nn->layers!=NULL);

    for (i=0;i<config->nlayers;i++)
    {
        if (i==0)
        {
            new_nn->layers[i]=neural_layer_create(config->neurons[i],
                config->signals,HIDDEN_NEURAL_LAYER);
            continue;
        }

        if (i+1==config->nlayers)
        {
            new_nn->layers[i]=neural_layer_create(config->neurons[i],
                config->neurons[i-1],OUTPUT_NEURAL_LAYER);
            continue;
        }

        new_nn->layers[i]=neural_layer_create(config->neurons[i],
            config->neurons[i-1],HIDDEN_NEURAL_LAYER);
    }
    
    return new_nn;
}





void neural_net_train(neural_net_t *nn,gsl_matrix *data)
{
    assert(nn!=NULL && data!=NULL);
    nn->config->train(nn,data);
    return;
}






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

