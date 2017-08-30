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




/*
 * @COMPLEXITY: Theta(1)
 *
 * The function hyperbolic_function() takes three immutable pointers
 * as parameters and casts them to pointers to doubles.This function
 * calculates the value of the hyperbolic tanget function given by
 * the following formula:
 *
 *          f(x) = ( 1 - e ^ - ( a * x + b ) ) / ( 1 + e ^ - ( a * x + b ) )
 *
 *  The hyperbolic tange function is continuous and fully differentiable
 *  at it's domain ( - oo, +oo ) with a range ( -1, 1 )
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double hyperbolic_function(const void *x,const void *a,const void *b)
{
    // Variable declarations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;

    // We calculate the numerator and denominator
    // seperately and then we perform the division
    // to retrieve the output value which we store
    // into the result variable.
    double numerator=1.0-exp(-((*aa)*(*xx)+(*bb)));
    double denominator=1.0+exp(-((*aa)*(*xx)+(*bb)));
    result=numerator/denominator; return result;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function hyperbolic_derivative() takes three immutable pointers
 * as parameters and casts them to pointers to doubles.This function
 * calculates the value of the first order derivative of the hyperbolic
 * tangent function given by the following formula:
 *
 *      f'(x) = ( 2 * a * e ^ - ( a * x + b ) ) / [ 1 + e ^ ( a * x + b ) ] ^ 2
 *
 *  The first order derivative of the hyperbolic tangent function.
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 * 
 */

double hyperbolic_derivative(const void *x,const void *a,const void *b)
{
    // Variable declarations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;

    // We calculate the numerator and denominator
    // seperately and then we perform the division
    // to retrieve the output value which we store
    // into the result variable.
    double numerator=2.0*(*aa)*exp(-((*aa)*(*xx)+*bb));
    double denominator=pow(1.0+exp(-((*aa)*(*xx)+*bb)),2);
    result=numerator/denominator; return result;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function logistic_function() takes three immutable pointers
 * as parameters and casts them to pointers to doubles.This function
 * calculates the value of the logistic function given by the following
 * formula:
 *      
 *      f(x) = 1 / ( 1 + e ^ - ( a * x + b ) )
 *
 *  The logistic function is continuous and fully differentiable at it's
 *  domain ( -oo, +oo ) with range ( 0, 1 )
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double logistic_function(const double *x,const double *a,const double *b)
{
    // Variable declarations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    
    // Calculating the output value of the logistic function.
    result=1.0/(1.0+exp(-((*aa)*(*xx)+*bb)));
    return result;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function logistic_derivative() takes three immutable pointers
 * as parameters and casts them to pointers to doubles.this function
 * calculates the value of the first order derivative of the logistic
 * function given by the following formula:
 *
 *      f'(x) = a * e ^ - ( a * x + b ) / ( 1 + e ^ - ( a * x + b ) ) ^ 2
 *  
 *  The first order derivative of the logigistic function.
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double logistic_derivative(const double *x,const double *a,const double *b)
{
    // Variable declarations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;
    
    // We calculate the numerator and denominator
    // seperately and then we perform the division
    // to retrieve the output value which we store
    // into the result variable.
    double numerator=(*aa)*exp(-((*aa)*(*xx)+(*bb)));
    double denominator=pow(1.0+exp(-((*aa)*(*xx)+(*bb))),2);
    result=numerator/denominator; return result;
}




/*
 * @COMPLEXITY: O(m*n)      where ( m x n ) are the dimensions of 
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
    // Variable declarations and initializations,
    // type assertions and type castings.
    double wij,vi; double temp,value;
    size_t i,j,l,s; double sum=0.0;
    assert(n!=NULL && v!=NULL);
    neural_net_t *nn=NULL; gsl_vector *vv=NULL;
    nn=(neural_net_t *)n; vv=(gsl_vector *)v;
    
    // Beginning the forward propagation process
    // by iterating through each layer of the network.
    for (l=0;l<nn->config->nlayers;l++)
    {
        // Retrieving the synaptic weights matrix,the
        // linear aggregators matrix and the signals output
        // matrix for the current neural layer data structure.
        gsl_matrix *W=neural_layer_getW(nn->layers[l]);
        gsl_matrix *I=neural_layer_getI(nn->layers[l]);
        gsl_matrix *Y=neural_layer_getY(nn->layers[l]);

        // If we are not at the first neural layer
        // retrieve the signals output matrix from
        // the previous neural layer.
        gsl_matrix *prevY=NULL;
        if (l>0) { prevY=neural_layer_getY(nn->layers[l-1]); }
        
        // Iterating over the synaptic weights
        // matrix data structure and the corresponding
        // input signals matrix and calculating the
        // linear aggregators and output signals.
        for (i=0;i<W->size1;i++)
        {
            // Calculating the linear aggregator
            // for the current i neuron in the layer
            // using the following formula:
            //
            //  if we are at the first neural layer
            //
            //          Ii = Sum ( W(i,j) * vv(j) )
            //
            //  Otherwise,for the rest neural layers
            //
            //          Ii = Sum ( W(i,j) * prevY(j,0) )
            // 
            sum=0.0;
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
                if (l==0) { vi=gsl_vector_get(vv,j); }
                else { vi=gsl_matrix_get(prevY,j,0); }
                sum+=wij*vi;
            }
            
            // Calculating the linear aggregator value
            // for the current i neuron in the l layer.
            gsl_matrix_set(I,i,0,sum);
            temp=gsl_matrix_get(I,i,0);

            // Fetching the above value into the activation function
            // and the result is stored into the corresponding output
            // signals matrix cell.We also make sure that we are not
            // at the final layer of the network,since each signals
            // output matrix at the hidden layers has an extra cell
            // containing the bias factor.
            value=nn->config->activate(&temp,&nn->config->alpha,&nn->config->beta);
            if (nn->config->nlayers==l+1) { s=i; } else { s=i+1; }
            gsl_matrix_set(Y,s,0,value);

        }
        
        // If we are at the hidden layers insert the bias factor -1
        // at the beginning of the signals output matrix.
        if (nn->config->nlayers>l+1) { gsl_matrix_set(Y,0,0,-1.0); }
    } return;
}




static void backward_propagate(const void *n,const void *v,const void *dv,const void *d)
{
    // Variable declarations and type assertions.
    double lij,wij,di,dk,wkj,yi,ii=0.0,temp;
    size_t l,i,j; double value,sum,temp_old;
    assert(n!=NULL && dv!=NULL && v!=NULL);
    gsl_vector *delta_prev=NULL,*delta_curr=NULL;
    gsl_matrix *W=NULL,*I=NULL,*Y=NULL,*L=NULL;
    gsl_matrix *Y_prev=NULL,*W_prev=NULL;
    gsl_vector_view delta_view;
    gsl_vector *delta_temp=NULL;
    
    // Type casting the given parameters to
    // their corresponding data structure type.
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_vector *dvv=NULL; dvv=(gsl_vector *)dv;
    gsl_vector *vv=NULL; vv=(gsl_vector *)v;
    gsl_vector *delta=NULL; delta=(gsl_vector *)d;

    
    // Beginning the backward propagation process
    // by starting the iteration from the output
    // layer all the way back to the input layer.
    for (l=nn->config->nlayers-1;l>0;l--)
    {
        W=neural_layer_getW(nn->layers[l]);
        I=neural_layer_getI(nn->layers[l]);
        Y=neural_layer_getY(nn->layers[l]);
        L=neural_layer_getL(nn->layers[l]);
        if (l>0) { Y_prev=neural_layer_getY(nn->layers[l-1]); }

        if (l+1==nn->config->nlayers)
        {
            delta_curr=dvv;
            //delta_view=gsl_vector_subvector(delta,0,I->size1);
            //delta_temp=(gsl_vector *)&delta_view;
            //gsl_vector_memcpy(delta_temp,delta_curr);
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
            W_prev=neural_layer_getW(nn->layers[l+1]);
            delta_curr=neural_layer_getD(nn->layers[l]);
            //delta_view=gsl_vector_subvector(delta,0,I->size1);
            //delta_temp=(gsl_vector *)&delta_view;
            //gsl_vector_memcpy(delta_temp,delta_curr);

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
            //temp_old=gsl_vector_get(delta_temp,i);
            //double sign=temp*temp_old,hta=nn->config->eta;
            for (j=0;j<W->size2;j++)
            {
                wij=gsl_matrix_get(W,i,j);
//                lij=gsl_matrix_get(L,i,j);
                
//                if (sign>0.0) { lij=(-1.0)*hta*lij; }
//                else if (sign<0.0) { lij=hta*lij; }

                if (l>0) { yi=gsl_matrix_get(Y_prev,j,0); }
                else { yi=gsl_vector_get(vv,j); }
                
//               if (temp>0.0) { wij-=lij*temp*yi; }
//                else if (temp<0.0) { wij+=lij*temp*yi; }
                wij+=nn->config->eta*temp*yi;
//                gsl_matrix_set(L,i,j,lij);
                gsl_matrix_set(W,i,j,wij);
            }
        }
    }
    
    return;
}

    



/*
 * @COMPLEXITY:
 *
 * The function resilient_backpropagation() takes two 
 * immutable pointers as arguments.The first one is cast 
 * into a neural_net_t pointer and the second one into
 * a gsl_matrix pointer.Once casting has been completed
 * The training samples are seperated from the desired
 * output samples using matrix views.Once the training
 * set X and desired output D have been defined the
 * training process for the neural network begins using
 * the resilient back-propagation algorithm.
 *
 * @param:  const void      *n
 * @param:  const void      *d
 * 
 */

void resilient_backpropagation(const void *n,const void *d)
{
    size_t max_row=0;
    assert(n!=NULL && d!=NULL); llint epoch_counter;
    size_t l,k1,k2,n1,n2,i; double err_curr,err_prev;
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_matrix *data=NULL; data=(gsl_matrix *)d;
    gsl_matrix *I_temp=NULL; gsl_vector *delta=NULL;
    k1=0; k2=0; n1=data->size1; n2=nn->config->signals;
    gsl_matrix_view X=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    k1=0; k2=nn->config->signals; n1=data->size1;
    n2=data->size2-nn->config->signals;
    gsl_matrix_view D=gsl_matrix_submatrix(data,k1,k2,n1,n2);     

    for (l=0;l<nn->config->nlayers;l++)
    {
        I_temp=neural_layer_getI(nn->layers[l]);
        if (I_temp->size1>max_row) { max_row=I_temp->size1; }
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
//        printf("%g,",fabs(err_curr-err_prev));
    } while (fabs(err_curr-err_prev)>nn->config->epsilon || epoch_counter<nn->config->epochs);
    gsl_vector_free(delta);
    return;
}

