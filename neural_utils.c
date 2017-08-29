/*
 * This file contains the defintions
 * of the procedures defined in the
 * neural_utils.h header file.
 *
 * @author: Endri Kastrati
 * @date:   27/08/2017
 *
 */



/*
 * Including the standard output library,
 * the standard utilities library,the standard
 * assertions library,the standard mathematics
 * library and the neural_net.h header file that
 * contains definitions of datatypes and function
 * prototypings regarding the neural network data
 * structure.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "neural_net.h"



double hyperbolic_function(const void *x,const void *a,const void *b)
{
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    double numerator=1.0-exp(-((*aa)*(*xx)+(*bb)));
    double denominator=1.0+exp(-((*aa)*(*xx)+(*bb)));
    result=numerator/denominator; return result;
}


double hyperbolic_derivative(const void *x,const void *a,const void *b)
{
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    double numerator=2.0*(*aa)*exp(-((*aa)*(*xx)+*bb));
    double denominator=pow(1.0+exp(-((*aa)*(*xx)+*bb)),2);
    result=numerator/denominator; return result;
}


double logistic_function(const double *x,const double *a,const double *b)
{
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    result=1.0/(1.0+exp(-((*aa)*(*xx)+*bb)));
    return result;
}



double logistic_derivative(const double *x,const double *a,const double *b)
{
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    double numerator=(*aa)*exp(-((*aa)*(*xx)+(*bb)));
    double denominator=pow(1.0+exp(-((*aa)*(*xx)+(*bb))),2);
    result=numerator/denominator; return result;
}




/*
 * @COMPLEXITY: O(m*n)      where m x n are the dimensions of 
 *                          the desired outputs matrix.
 * 
 * The function mean_square_error_calculate() takes three immutable
 * pointers as arguments.The first one is cast into a neural_net_t
 * pointer,the second into a gsl_matrix pointer and the third one
 * into a long long int pointer.Once the casting has been completed
 * the mean squared error is calculated.The squared error function
 * is employed to measure the local performance associated with the
 * results produced by the output neurons with respect to the given
 * samples.
 *
 * @param:  const void      *n
 * @param:  const void      *d
 * @param:  const void      *p
 * @return: double
 *
 */

double mean_square_error_calculate(const void *n,const void *d,const void *p)
{
    // temporary variable declarations
    // and assertions about the parameters.
    double di,yi;
    size_t i,j; double sum=0.0,total_error=0.0;
    assert(n!=NULL && d!=NULL && p!=NULL);
    
    // Casting the given parameters into their
    // corresponding datatype pointer.
    llint *nsamples=NULL; nsamples=(llint *)p;
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_matrix *D=NULL,*Y=NULL; D=(gsl_matrix *)d;

    // Retrieving the signal outputs matrix for the
    // output layer of the neural network.
    Y=neural_layer_getY(nn->layers[nn->config->nlayers-1]);
    
    
    // Beginning the mean squared error
    // calculation procedure.
    for (i=0;i<D->size1;i++)
    {
        // First we calculate the squared
        // error for each output neuron and
        // sum them together.
        sum=0.0;
        for (j=0;j<D->size2;j++)
        {
            di=gsl_matrix_get(D,i,j);
            yi=gsl_matrix_get(Y,j,0);
            sum+=pow(di-yi,2);
        }
        
        // Once summation has been completed
        // we divide it by two and add it into
        // the total error variable.
        total_error+=(double )sum/(double )2.0;
    }
    
    // To get the measurement for the global performance
    // of the training algorithm we calculate the mean
    // squared error by dividing the total erro by the
    // total number of training samples available.
    total_error=(double )total_error/(double )(*nsamples);
    return total_error;
}



/*
 * @COMPLEXITY: O(l*m*n)    Where l is the number of layers
 *                          in the neural network and the
 *                          ( m x n ) the dimensiosn of the
 *                          largest synaptic weights marix.
 * 
 * The static function forward_propagate() takes two immutable
 * void pointers as parameters and casts the first one into
 * a neural_net_t pointer and the second one into a gsl_vector 
 * pointer.Once the casting has been completed the given vector
 * is fetched as input signals into the neural network.We start
 * by forward propagating the corresponding inputs and outputs 
 * at each layer of the neural network until we reach the output
 * layer.Once the output layer has been reached the corresponding
 * output signals have been estimated.
 *
 * @param:  const void      *n
 * @param:  const void      *v
 * @return: void
 *
 */

static void forward_propagate(const void *n,const void *v)
{
    double wij,vi; double temp,value;
    size_t i,j,l,s; double sum=0.0;
    assert(n!=NULL && v!=NULL);
    neural_net_t *nn=NULL; gsl_vector *vv=NULL;
    nn=(neural_net_t *)n; vv=(gsl_vector *)v;

    for (l=0;l<nn->config->nlayers;l++)
    {


        gsl_matrix *W=neural_layer_getW(nn->layers[l]);
        gsl_matrix *I=neural_layer_getI(nn->layers[l]);
        gsl_matrix *Y=neural_layer_getY(nn->layers[l]);
        gsl_matrix *prevY=NULL;
        if (l>0) { prevY=neural_layer_getY(nn->layers[l-1]); }

        for (i=0;i<W->size1;i++)
        {
            sum=0.0;
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
                if (l==0) { vi=gsl_vector_get(vv,j); }
                else { vi=gsl_matrix_get(prevY,j,0); }
                sum+=wij*vi;
            }

            gsl_matrix_set(I,i,0,sum);
            temp=gsl_matrix_get(I,i,0);
            value=nn->config->activate(&temp,&nn->config->alpha,&nn->config->beta);
            if (nn->config->nlayers==l+1) { s=i; } else { s=i+1; }
            gsl_matrix_set(Y,s,0,value);

        }
        
        if (nn->config->nlayers>l+1) { gsl_matrix_set(Y,0,0,-1.0); }
    } return;
}




static void backward_propagate(const void *n,const void *v,const void *dv,const void *d)
{
    double wij,di,dk,wkj,yi,ii=0.0,temp;
    size_t l,i,j; double value,sum;
    assert(n!=NULL && dv!=NULL && v!=NULL);
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_vector *dvv=NULL; dvv=(gsl_vector *)dv;
    gsl_vector *vv=NULL; vv=(gsl_vector *)v;
    gsl_vector *delta=NULL;

    delta=(gsl_vector *)d;
    gsl_vector_view delta_view1,delta_view2;
    gsl_vector *delta_prev=NULL,*delta_curr=NULL;
    


    for (l=nn->config->nlayers-1;l>0;l--)
    {
        gsl_matrix *W=neural_layer_getW(nn->layers[l]);
        gsl_matrix *I=neural_layer_getI(nn->layers[l]);
        gsl_matrix *Y=neural_layer_getY(nn->layers[l]);
        gsl_matrix *Y_prev=NULL;
        if (l>0) { Y_prev=neural_layer_getY(nn->layers[l-1]); }

        if (l+1==nn->config->nlayers)
        {
            delta_view1=gsl_vector_subvector(delta,0,dvv->size);
            delta_curr=(gsl_vector *)&delta_view1;

            for (i=0;i<delta_curr->size;i++)
            {
                yi=gsl_matrix_get(Y,i,0);
                di=gsl_vector_get(dvv,i);
                ii=gsl_matrix_get(I,i,0);
                value=nn->config->derivative(&ii,&nn->config->alpha,&nn->config->beta);
                temp=(di-yi)*value; gsl_vector_set(delta_curr,i,temp);
            }

            delta_prev=delta_curr;
        }
        else
        {
            gsl_matrix *W_prev=neural_layer_getW(nn->layers[l+1]);
            delta_view2=gsl_vector_subvector(delta,0,I->size1);
            delta_curr=(gsl_vector *)&delta_view2;

            for (i=0;i<delta_curr->size;i++)
            {
                sum=0.0;
                for (j=0;j<W_prev->size1;j++)
                {
                    dk=gsl_vector_get(delta_prev,j);
                    wkj=gsl_matrix_get(W_prev,j,i);
                    sum+=dk*wkj;
                }

                ii=gsl_matrix_get(I,i,0);
                value=nn->config->derivative(&ii,&nn->config->alpha,&nn->config->beta);
                temp=-sum*value; 
                gsl_vector_set(delta_curr,i,temp);
            }
            
            delta_prev=delta_curr;
        }

        for (i=0;i<W->size1;i++)
        {
            temp=gsl_vector_get(delta_curr,i);
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
                if (l>0) { yi=gsl_matrix_get(Y_prev,j,0); }
                else { yi=gsl_vector_get(vv,j); }
                wij+=nn->config->eta*temp*yi;
                gsl_matrix_set(W,i,j,wij);
            }
        }
    }
    
    return;
}

    



/*
 * @COMPLEXITY:
 *
 * The function backpropagation() takes two immutable
 * pointers as arguments.The first one is cast into
 * a neural_net_t pointer and the second one into
 * a gsl_matrix pointer.Once casting has been completed
 * The training samples are seperated from the desired
 * output samples using matrix views.Once the training
 * set X and desired output D have been defined the
 * training process for the neural network begins.
 *
 * @param:  const void      *n
 * @param:  const void      *d
 *
 * 
 */

void backpropagation(const void *n,const void *d)
{
    size_t max_row=0;
    assert(n!=NULL && d!=NULL); llint epoch_counter;
    size_t l,k1,k2,n1,n2,i; double err_curr,err_prev;
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_matrix *data=NULL; data=(gsl_matrix *)d;
    k1=0; k2=0; n1=data->size1; n2=nn->config->signals;
    gsl_matrix_view X=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    k1=0; k2=nn->config->signals; n1=data->size1;
    n2=data->size2-nn->config->signals;
    gsl_matrix_view D=gsl_matrix_submatrix(data,k1,k2,n1,n2);     
    gsl_vector *delta=NULL;
    
    for (l=0;l<nn->config->nlayers;l++)
    {
        gsl_matrix *temp_I=neural_layer_getI(nn->layers[l]);
        if (max_row<temp_I->size1) { max_row=temp_I->size1; }
    }
    
    delta=gsl_vector_alloc(max_row);


    epoch_counter=0;
    do
    {
        err_prev=mean_square_error_calculate(nn,&D,&data->size1);
        for (i=0;i<data->size1;i++)
        {
            gsl_matrix *XX=NULL; XX=(gsl_matrix *)&X;
            gsl_vector_view vector_input_row=gsl_matrix_row(XX,i);
            forward_propagate(nn,&vector_input_row); 
            gsl_matrix *DD=NULL; DD=(gsl_matrix *)&D;
            gsl_vector_view vector_output_row=gsl_matrix_row(DD,i);
            backward_propagate(nn,&vector_input_row,&vector_output_row,delta);
        }

        err_curr=mean_square_error_calculate(nn,&D,&data->size1); epoch_counter+=1;
    } while (fabs(err_curr-err_prev)>nn->config->epsilon || epoch_counter<nn->config->epochs);
    gsl_vector_free(delta);
    return;
}

