/*
 * This file contains definitions of 
 * procedures regarding the dataset_t
 * data structure.
 *
 * @author: Endri Kastrati
 * @date:   11/09/2017
 *
 */



/*
 * Including the standard utilities library,
 * the standard assertions library,the standard
 * string manipulation library and the header
 * file dataset.h that contains datatype definitions
 * and function prototypings regarding the dataset
 * data structure.
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "dataset.h"





/*
 * @COMPLEXITY: O(m*n)      Where ( m x n) are the dimensions
 *                          of the read dataset values.
 *
 * The function dataset_create() takes four arguments as parameters.
 * The first argument is a file stream,the second argument is the
 * type and the last two are function pointers to scaling and descaling
 * functions.This function reads the dimensions of the dataset from the
 * first and second line and then loads the dataset values into a gsl_matrix
 * data structure.It also ensures that the bias column is inserted at the
 * first column of the matrix.Also everything has been loaded and properly
 * formatted the newly instantiated dataset_t data structure is returned.
 *
 * @param:  FILE        *f
 * @param:  int         type
 * @param:  ScalerFn    scaler
 * @param:  ScalerFn    descaler
 * @return: dataset_t   *
 *
 */

dataset_t *dataset_create(FILE *f,int type,ScalerFn scaler,ScalerFn descaler)
{
    // variable declarations
    // type assertions and
    // verifications.
    char data[100]; char *flag;
    assert(f!=NULL); size_t i;
    assert(type==DATASET_CLASSIFY || type==DATASET_PREDICT);
    
    // allocating memory from the heap for a newly
    // instance of the dataset data structure.
    dataset_t *new_dataset=NULL; FILE *stream=f;
    new_dataset=(dataset_t *)malloc(sizeof(*new_dataset));
    assert(new_dataset!=NULL);

    // Initializing the components of the
    // dataset to their corresponding values.
    new_dataset->type=type;
    new_dataset->scaler=scaler;
    new_dataset->descaler=descaler;
    new_dataset->maximums=NULL;
    new_dataset->minimums=NULL;
    
    // Reading the row dimensions from the opened stream.If something
    // goes wrong an error is printed into standard error stream and
    // program execution is immediately terminated.
    if ((flag=fgets(data,100,stream))!=NULL) { new_dataset->rows=atoi(data); }
    else { fprintf(stderr,"Could not read the number of rows.\n"); exit(EXIT_FAILURE); }
    
    // Reading the column dimensions from the opened stream.If something
    // goes wrong an error is printed into the standard error stream and
    // the program execution is immediately terminated.
    if ((flag=fgets(data,100,stream))!=NULL) { new_dataset->columns=atoi(data); }
    else { fprintf(stderr,"Could not read the number of columns.\n"); exit(EXIT_FAILURE); }
    
    // Allocating a new gsl_matrix data structure based on the read dimensions
    // and then reading the dataset values from the stream and storing them into
    // the corresponding matrix cell.
    new_dataset->data=gsl_matrix_alloc(new_dataset->rows,new_dataset->columns+1); 
    gsl_matrix_view view=gsl_matrix_submatrix(new_dataset->data,0,1,new_dataset->rows,new_dataset->columns);
    gsl_matrix_fscanf(stream,(gsl_matrix *)&view);

    // Inserting the bias factor for all rows at the first column
    // and returning the address of the newly instantiated dataset.
    for (i=0;i<new_dataset->rows;i++) { gsl_matrix_set(new_dataset->data,i,0,-1.0); }
    new_dataset->columns+=1;
    return new_dataset;
}




/*
 * @COMPLEXITY: O(n)    Where n is the total number of columns.
 * 
 * The function dataset_dump_minmax() takes two arguments
 * as parameters.The first argument is a dataset_t data
 * structure and the second one is the name of a directory.
 * This function saves the mininum and maximum components
 * of the dataset into binary in a file called "minmax.bin"
 *
 * @param:  dataset     *ds
 * @param:  char        *directory
 * @return: void
 *
 */

void dataset_dump_minmax(dataset_t *ds,char *directory)
{
    FILE *f=NULL;
    int len1,len2;
    char *filepath=NULL;
    char *filename="/minmax.bin";
    assert(ds!=NULL && directory!=NULL);
    len1=strlen(directory);
    len2=strlen(filename);
    filepath=(char *)malloc((len1+len2+1)*sizeof(char ));
    assert(filepath!=NULL);
    strcpy(filepath,directory);
    strcat(filepath,filename);
    f=fopen(filepath,"wb");
    fwrite(&ds->columns,sizeof(long long int ),1,f);
    gsl_vector_fwrite(f,ds->minimums);
    gsl_vector_fwrite(f,ds->maximums);
    fclose(f); f=NULL; free(filepath);
    return;
}



/*
 * @COMPLEXITY: O(n)    Where n is the total number of columns.
 *
 * The function dataset_load_minmax() takes two arguments as
 * parameters.The first argument is a dataset_t data structure
 * and the second one is the name of a directory.This function
 * loads the binary data from the "minmax.bin" file that is
 * located within the specified directory.
 *
 * @param:  dataset_t   *ds
 * @param:  char        *directory
 * @return: void
 *
 */

void dataset_load_minmax(dataset_t *ds,char *directory)
{
    FILE *f=NULL; int dummy=0.0;
    int len1,len2; dummy+=0.0;
    long long int columns;
    char *filepath=NULL;
    char *filename="/minmax.bin";
    assert(ds!=NULL && directory!=NULL);
    len1=strlen(directory);
    len2=strlen(filename);
    filepath=(char *)malloc((len1+len2+1)*sizeof(char ));
    assert(filepath!=NULL);
    strcpy(filepath,directory);
    strcat(filepath,filename);
    f=fopen(filepath,"rb");
    dummy=fread(&columns,sizeof(long long int ),1,f);
    ds->minimums=gsl_vector_alloc(columns);
    ds->maximums=gsl_vector_alloc(columns);
    gsl_vector_fread(f,ds->minimums);
    gsl_vector_fread(f,ds->maximums);
    fclose(f); f=NULL; free(filepath);
    return;
}




/*
 * @COMPLEXITY: O(m*n)  Where ( m x n ) are the dimensions
 *                      of the dataset matrix.
 * 
 * The function dataset_scale() takes only one argument
 * as parameter,namely a dataset_t data structure and
 * normalizes the components of each column in the dataset
 * matrix based on the given scaler function and the minimum
 * and maximum value of the corresponding column.
 *
 * @param:  dataset_t   *ds
 * @return: void
 *
 */

void dataset_scale(dataset_t *ds)
{
    // Variable declarations
    // type assertions and
    // default instantiations.
    int flag1=0,flag2=0;
    double a,b,value; gsl_vector_view v;
    assert(ds!=NULL); size_t i,j;
    double max=0.0,min=0.0;
    
    // Checking if the maximums vector has
    // been defined and if not we allocate
    // a new one.
    if (ds->maximums==NULL)
    {
        ds->maximums=gsl_vector_calloc(ds->columns);
        assert(ds->maximums!=NULL); flag1=1;
    }
    
    // Checking if the minimums vector has
    // been defined and if not we allocate
    // a new one.
    if (ds->minimums==NULL)
    {
        ds->minimums=gsl_vector_calloc(ds->columns);
        assert(ds->minimums!=NULL); flag2=1;
    }
    
    // Checking the dataset type and based on it
    // we declare the a,b values for the scaling 
    // function.If we are doing classification then
    // we want to scale between [0,1] and therefore
    // a is set to 1 and b is set to 0.If we are doing
    // prediction (regression) a is set to 2 and b is
    // set to 1.
    if (ds->type==DATASET_CLASSIFY) { a=1.0; b=0.0; }
    if (ds->type==DATASET_PREDICT)  { a=2.0; b=1.0; }
    

    // Iterating over the columns of the dataset
    // matrix and applying the normalization function
    // that was given as a function pointer during the
    // instantiation of the dataset data structure.
    for (j=1;j<ds->columns;j++)
    {
        v=gsl_matrix_column(ds->data,j);
        min=gsl_vector_min((gsl_vector *)&v);
        max=gsl_vector_max((gsl_vector *)&v);
        if (flag1==1 && flag2==1)
        {
            gsl_vector_set(ds->minimums,j,min);
            gsl_vector_set(ds->maximums,j,max);
        }
        else
        {
            min=gsl_vector_get(ds->minimums,j);
            max=gsl_vector_get(ds->maximums,j);
        }
        
        for (i=0;i<ds->rows;i++)
        {
            value=gsl_vector_get((gsl_vector *)&v,i);
            value=ds->scaler(min,max,value,a,b);
            gsl_vector_set((gsl_vector *)&v,i,value);
        }
    } return;
}




/*
 * @COMPLEXITY: Theta(1)
 *
 * The function dataset_free() takes a dataset_t
 * data structure as an argument and deallocates
 * all memory blocks associated with it and it's
 * components.
 *
 * @param:  dataset_    *ds
 * @return: void
 *
 */

void dataset_free(dataset_t *ds)
{
    assert(ds!=NULL);
    if (ds->maximums!=NULL) { gsl_vector_free(ds->maximums); }
    if (ds->maximums!=NULL) { gsl_vector_free(ds->minimums); }
    gsl_matrix_free(ds->data); free(ds); ds=NULL;
    return;
}
