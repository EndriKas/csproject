
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "dataset.h"





dataset_t *dataset_create(FILE *f,int type,ScalerFn scaler,ScalerFn descaler)
{
    char data[100]; char *flag;
    assert(f!=NULL); size_t i;
    assert(type==DATASET_CLASSIFY || type==DATASET_PREDICT);
    dataset_t *new_dataset=NULL; FILE *stream=f;
    new_dataset=(dataset_t *)malloc(sizeof(*new_dataset));
    assert(new_dataset!=NULL); new_dataset->type=type;
    new_dataset->scaler=scaler;
    new_dataset->descaler=descaler;
    new_dataset->maximums=NULL;
    new_dataset->minimums=NULL;
    if ((flag=fgets(data,100,stream))!=NULL) { new_dataset->rows=atoi(data); }
    else { fprintf(stderr,"Could not read the number of rows.\n"); exit(EXIT_FAILURE); }
    if ((flag=fgets(data,100,stream))!=NULL) { new_dataset->columns=atoi(data); }
    else { fprintf(stderr,"Could not read the number of columns.\n"); exit(EXIT_FAILURE); }
    new_dataset->data=gsl_matrix_alloc(new_dataset->rows,new_dataset->columns+1); 
    gsl_matrix_view view=gsl_matrix_submatrix(new_dataset->data,0,1,new_dataset->rows,new_dataset->columns);
    gsl_matrix_fscanf(stream,(gsl_matrix *)&view);
    for (i=0;i<new_dataset->rows;i++) { gsl_matrix_set(new_dataset->data,i,0,-1.0); }
    new_dataset->columns+=1;
    return new_dataset;
}



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



void dataset_scale(dataset_t *ds)
{
    int flag1=0,flag2=0;
    double a,b,value; gsl_vector_view v;
    assert(ds!=NULL); size_t i,j;
    double max=0.0,min=0.0;
    
    if (ds->maximums==NULL)
    {
        ds->maximums=gsl_vector_calloc(ds->columns);
        assert(ds->maximums!=NULL); flag1=1;
    }

    if (ds->minimums==NULL)
    {
        ds->minimums=gsl_vector_calloc(ds->columns);
        assert(ds->minimums!=NULL); flag2=1;
    }

    if (ds->type==DATASET_CLASSIFY) { a=1.0; b=0.0; }
    if (ds->type==DATASET_PREDICT)  { a=2.0; b=1.0; }

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



void dataset_free(dataset_t *ds)
{
    assert(ds!=NULL);
    if (ds->maximums!=NULL) { gsl_vector_free(ds->maximums); }
    if (ds->maximums!=NULL) { gsl_vector_free(ds->minimums); }
    gsl_matrix_free(ds->data); free(ds); ds=NULL;
    return;
}
