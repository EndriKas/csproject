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
#include "neural_utils.h"
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

double logistic_function(const void *x,const void *a,const void *b)
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
 *  The first order derivative of the logistic function.
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double logistic_derivative(const void *x,const void *a,const void *b)
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
 * @COMPLEXITY: Theta(1)
 *
 * The function linear_function() takes three immutable pointers 
 * as parameters and casts them to pointers to doubles.This function
 * calculates the value of the linear function given by the following
 * formula:
 *
 *      f(x) = a * x + b
 *
 *  The linear function is continuous and fully differentiable at it's
 *  domain ( -oo, +oo ) with range ( -oo, +oo )
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double linear_function(const void *x,const void *a,const void *b)
{
    // Variable declrations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *xx=NULL,*aa=NULL,*bb=NULL,result=0.0;
    xx=(double *)x; aa=(double *)a; bb=(double *)b;

    // We calculate the value of the linear function
    // for the given parameters and return the result.
    result=(*aa)*(*xx)+(*bb); return result;
}



/*
 * @COMPLEXITY: Theta(1)
 *
 * The function linear_derivative() takes three immutable pointers
 * as parameters and casts them to pointers to doubles.This function
 * calculates the value of the first order derivative of the linear
 * function given by the following formula:
 *
 *      f(x) = a
 *
 *  The first order derivative of the linear function.
 *
 *  @param:     const void      *x
 *  @param:     const void      *a
 *  @param:     const void      *b
 *  @return:    double
 *
 */

double linear_derivative(const void *x,const void *a,const void *b)
{
    // Variable declarations,type assertions
    // and castings to doubles.
    assert(x!=NULL && a!=NULL && b!=NULL);
    double *aa=NULL,result=0.0; aa=(double *)a; 

    // We calculate the value of the first order
    // derivative of the linear function and return
    // the results.
    result=(*aa); return result;
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
    size_t *nsamples=NULL; nsamples=(size_t *)p;
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




/*
 * @COMPLEXITY: 
 *
 * The function backward_propagate() takes three immutable pointers as parameters.
 * The first parameter is cast into a neural_net_t data structure.The second and
 * the third one are cast into gsl_vector data structures.This function iterates
 * backwards,namely from the output layer to the input layer and adjusts the 
 * synaptic weights of each layer.First the adjusted synaptic weights of the
 * neurons of the output layer are calculated by comparing the deviation of
 * the produced responses and the corresponding desired values.Secondly this 
 * error back-propagates to the neurons of the previous layer weighted by the
 * values of the synaptic weights that were previously adjusted in all the
 * posterior layers.Consequently,the desired response of a neuron in a hidden
 * layer must be determined with respect to the neurons that are directly
 * connected to it and that have been already adjusted in the previous stage.
 *
 * @param:  const void      *n
 * @param:  const void      *in
 * @param:  const void      *out
 * @return: void
 *
 */

static void backward_propagate(const void *n,const void *in,const void *out)
{
    // Variable declarations and type assertions.
    // Most of the declared variables have been
    // named in such a way as to provide a detailed
    // walkthrough of the back-propagate procedure.
    llint l; size_t k,j,i; double wji,wji_o,shift;
    double yj,dj,value,ij,yi,wkj,dk,sum;
    assert(n!=NULL && in!=NULL && out!=NULL);
    gsl_matrix *W=NULL; gsl_matrix *I=NULL;
    gsl_matrix *Y=NULL; gsl_matrix *D=NULL;
    gsl_matrix *O=NULL; gsl_matrix *prevY=NULL;
    gsl_matrix *postW=NULL; gsl_matrix *postD=NULL;
    

    // Casting the first parameter into a neural_net_t
    // data structure and the second and third into a
    // gsl_vector data structure.
    neural_net_t *nn=NULL; nn=(neural_net_t *)n;
    gsl_vector *input=NULL;input=(gsl_vector *)in;
    gsl_vector *output=NULL; output=(gsl_vector *)out;
    
    
    // Beginning the iteration from the output layer
    // all the way to the input layer of the neural net.
    for (l=nn->config->nlayers-1;l>=0;l--)
    {
        // Retrieving the synaptic weights matrix
        // for the current layer.The components of
        // this matrix are adjusted during this process.
        W=neural_layer_getW(nn->layers[l]);

        // Retrieving the linear aggregators matrix for
        // the current layer.This matrix will be used 
        // during the adjustment of the synaptic weights.
        I=neural_layer_getI(nn->layers[l]);

        // Retrieving the signals output matrix for the
        // current layer.This matrix will also be used
        // during the adjustment of the synaptic weights.
        Y=neural_layer_getY(nn->layers[l]);

        // Retrieving the local gradient matrix for the
        // current layer.This matrix will be used during
        // the adjustment of the synaptic weights.
        D=neural_layer_getD(nn->layers[l]);

        // Retrieving the synaptic weights matrix from
        // the previous training epoch.This matrix will
        // be used for the optimization technique aka
        // momentum parameter optimization.
        O=neural_layer_getO(nn->layers[l]);

        
        // If we are not at the input layer, retrieve the 
        // signals output matrix from the previous layer.
        if (l>0) { prevY=neural_layer_getY(nn->layers[l-1]); }


        // There are two main stage to the back-propagation
        // process.The first stage concerns the adjustment
        // of the output layer which is done using the given
        // desired output signals.The second stage concerns
        // the adjustment of the intermediate layers which
        // do not have access to the desired values for outputs.
        if (l+1==nn->config->nlayers)
        {
            // iterating over the elements of the
            // current local gradient matrix.
            for (j=0;j<D->size1;j++)
            {
                // The calculation for the local gradient
                // related to the jth neuron in the output
                // layer is given by the following formula:
                //
                //      delta(j) = ( output(j) - Y(j) ) * g'(I(j)) 
                //
                // Where g is the derivative of the activation function.
                // The components of the gradient matrix are overwritten
                // in-place without needing the allocation of extra memory.
                yj=gsl_matrix_get(Y,j,0);
                dj=gsl_vector_get(output,j);
                ij=gsl_matrix_get(I,j,0);
                value=nn->config->derivative(&ij,&nn->config->alpha,&nn->config->beta);
                gsl_matrix_set(D,j,0,(dj-yj)*value);
            }
        }
        else
        {
            // For the calculation of the local gradient
            // matrix for the hidden layers we are going
            // to require the synaptic weights matrix and
            // the local gradient matrix of the posterior
            // neural layer of the network.
            postW=neural_layer_getW(nn->layers[l+1]);
            postD=neural_layer_getD(nn->layers[l+1]); 

            // Iterating over the elements of the
            // current local gradient matrix.
            for (j=0;j<D->size1;j++)
            {
                // Before we calculate the local gradient related to the jth neuron,
                // first we have to sum up the multiplication of the local gradients
                // and synaptic weights of all neurons of the posterior layer that
                // are connected to the current jth neuron of the current layer.
                sum=0.0;
                for (k=1;k<postW->size1;k++)
                {
                    // The summation is calculated using the 
                    // following formula:
                    //
                    //      Sum = Sum + posterior_delta(k)*posterior_W(k,j)
                    //
                    // We iterate over all neurons connected to the current
                    // jth neuron and aggregate the expected desired output value.
                    wkj=gsl_matrix_get(postW,k,j);
                    dk=gsl_matrix_get(postD,k,0);
                    sum+=wkj*dk;
                }
                
                // To calculate the local gradient related to the jth neuron
                // of the current layer we use sum obtained above and the
                // following formula:
                //
                //      delta(j) = - ( sum ) * g'(I(j))
                // 
                // Where g' is the derivative of the activaion function.
                // The components of the gradient matrix are overwritten
                // in-place without needing the allocation of extra memory.
                ij=gsl_matrix_get(I,j,0);
                value=nn->config->derivative(&ij,&nn->config->alpha,&nn->config->beta);
                gsl_matrix_set(D,j,0,-sum*value);
            }
        }
        
        // Once we have calculate the corresponding local gradient
        // matrix,it is time to adjust the synaptic weights of the
        // current neural layer.We begin iterating over the rows of
        // the synaptic weights matrix for the current layer.
        for (j=0;j<W->size1;j++)
        {
            // Retrieve the current local gradient
            // related to the jth neuron of the current
            // neural layer and iterating over the columns
            // of the synaptic weights matrix.
            dj=gsl_matrix_get(D,j,0);
            for (i=0;i<W->size2;i++)
            {
                // Retrieving the current value of the synaptic
                // weights matrix cell and it's value from the
                // previous training epoch.
                wji=gsl_matrix_get(W,j,i);
                wji_o=gsl_matrix_get(O,j,i);

                // Checking if we are at the hidden layers or
                // we have reach the input layer.If we are at
                // the hidden layers we use the output signals
                // of the previous layer otherwise we use the
                // given input signals.
                if (l>0)    { yi=gsl_matrix_get(prevY,i,0); }
                else        { yi=gsl_vector_get(input,i);   }
                
                // Calculating the total amount the current synaptic
                // weights cell is going to be shifted by the following
                // formula:
                //
                //      W(j,i) = W(j,i) + momentum * ( W(j,i) - O(j,i) ) + hta * delta(j) * Y(i)
                // 
                // Once the shift has been calculated the cell is updated and
                // so previous value written into the O matrix.
                shift=wji+0.09*(wji-wji_o)+(nn->config->eta)*dj*yi;
                gsl_matrix_set(O,j,i,wji);
                gsl_matrix_set(W,j,i,shift);
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
 * @return: void
 * 
 */

void backpropagation(const void *n,const void *d)
{
    // Variable declarations and initializations
    // and type verifications.
    size_t k1,k2,n1,n2,i; 
    assert(n!=NULL && d!=NULL); llint epoch_counter=0;
    double err_curr=1.0,err_prev=1.0,loss=0.0;
    neural_net_t *nn=NULL; gsl_matrix *data=NULL;
    gsl_vector_view vector_input_row;
    gsl_vector_view vector_output_row;

    
    // Casting the given parameters into the corresponding
    // datatypes.The first one into a neural_net_t data
    // structure and the second one into a gsl_matrix.
    nn=NULL; nn=(neural_net_t *)n;
    data=NULL; data=(gsl_matrix *)d;
    

    // Retrieving a matrix view of the given dataset
    // for the signals input only.More specifically
    // the X component of the dataset.
    k1=0; k2=0; n1=data->size1; n2=nn->config->signals;
    gsl_matrix_view X=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    

    // Retrieving a matrix view of the given dataset
    // for the desired output only.More specifically
    // the Y component of the dataset.
    k1=0; k2=nn->config->signals; n1=data->size1;
    n2=data->size2-nn->config->signals;
    gsl_matrix_view D=gsl_matrix_submatrix(data,k1,k2,n1,n2);
    
    // Beginning the training process of the back-propagation
    // algorithm.We stop the procedure when the maximum epoch
    // limit or convergence limit has beenr reached. 
    do
    {
        // Calculate the current mse value and begin
        // iterating over the given training dataset.
        err_prev=mean_square_error_calculate(nn,&D,&data->size1);
        for (i=0;i<data->size1;i++)
        {
            // Get the ith input row and fetch it into the
            // neural network using the forward_propagate procedure.
            vector_input_row=gsl_matrix_row((gsl_matrix *)&X,i);
            forward_propagate(nn,&vector_input_row);
            
            // Get the ith output row and fetch it into the
            // neural network using the backward propagate procedure.
            vector_output_row=gsl_matrix_row((gsl_matrix *)&D,i);
            backward_propagate(nn,&vector_input_row,&vector_output_row);
        }
        
        // Retrieve the mean square error value after the current training
        // epoch and increment the epoch counter by one.Print the epoch
        // counter,current loss and the current mean square error into the
        // standard output stream.
        err_curr=mean_square_error_calculate(nn,&D,&data->size1); epoch_counter+=1; loss=fabs(err_curr-err_prev);
        printf(CYN"EPOCHS"RESET" = %lld, "RED"LOSS"RESET" = %g, "YEL"MSE"RESET" = %g\n",epoch_counter,loss,err_curr);
    } while (loss>nn->config->epsilon && epoch_counter<nn->config->epochs);
    return;
}

