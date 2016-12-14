#include <stdio.h>
#include <stdlib.h>
#include <math.h>
float ** load_data(char * nom_fichier, int *nb_variable , int *nb_exemple, int *nb_classe)
{
    FILE *f =fopen(nom_fichier, "r");

    fscanf(f,"%d %d %d",nb_exemple,nb_variable,nb_classe);
    float ** my_data=(float **)malloc ((*nb_exemple)*sizeof(float *));
    int i, j;
    for (i=0;i<*nb_exemple;i++)
    {
       my_data[i]=(float *)malloc(((*nb_variable)+1)*sizeof (float));
        for (j=0;j<*nb_variable;j++)
        {
            fscanf(f,"%f",&my_data[i][j]);
        }
        fscanf(f,"%f",&my_data[i][j]);
    }
    return my_data;
}


float write_data_lib_svm (float **data, char *nom_fichier,int nb_variable , int nb_exemple)
{
    FILE *f= fopen(nom_fichier,"w");

    if (f!=NULL)
    {
        int i;
        for (i=0;i<nb_exemple;i++)
        {
           fprintf(f,"%d ",(int)data[i][nb_variable]);
            int j;
            for (j=0;j<nb_variable;j++)
            {
                fprintf (f,"%d:%f ",j+1,data[i][j]);
            }
            fprintf(f,"\n");
            printf("%d \n",i);
        }
        fclose(f);
    }
}

float dot_product_2_parzen (float *mon_vec, int taille,float h)
{
    float dot_product=0;
    int i;
    for (i=0;i<taille;i++)
    {
        dot_product+=(mon_vec[i]*mon_vec[i]);
    }
    return dot_product;
}

float gaussian_kernel_parzen(float  **Base_app, float  **Base_test, int ex_base_1,int ex_base_2,int nb_variable,float h)
{
int i;
float vect[nb_variable];

for (i=0;i<nb_variable;i++)
{
vect[i]=(Base_app[ex_base_1][i]-Base_test[ex_base_2][i])/h;
}
 return exp(dot_product_2_parzen(vect,nb_variable,h)/-2)/sqrt(2*M_PI);

}




float parzen_window_estimate(float  **Base_app, float  **Base_test, int exemple,int classe,int nb_exemple,int nb_variable,float h)
{
int i=0;
float prob_classe=0;
for (i=0;i<nb_exemple;i++)
{
if( Base_app[i][nb_variable]==classe)
{
prob_classe+=(gaussian_kernel_parzen(Base_app,Base_test,i,exemple,nb_variable,h)/(nb_exemple*h));
}
}
return prob_classe;
}


int * parzen_classification(float **base_app, float **base_test, int nb_variable, int nb_classe, int nb_exemple_test,int nb_exemple_app,float h)
{
    int *result=(int *)malloc (nb_exemple_test*sizeof (int));
    int i;
    for (i=0;i<nb_exemple_test;i++)
    {

        float max_prob_classe=-1;
        int classe=1;
        int  j=0;
        for (j=1;j<nb_classe+1;j++)
        {
            float prob_tmp;
            if( (prob_tmp=parzen_window_estimate(base_app,base_test,i,j,nb_exemple_app,nb_variable,h))>max_prob_classe)
            {
                max_prob_classe=prob_tmp;
                classe=j;
            }
        }
         // printf("%d %d\n",i,classe);
        result[i]=classe;
    }
    return result;

}





int main(int argc , char **argv)
{
    int nb_exemple_app,nb_variable_app,nb_classe_app;
    int nb_exemple_test,nb_variable_test,nb_classe_test;
    float **base_app=load_data(argv[1],&nb_variable_app,&nb_exemple_app,&nb_classe_app);
    float **base_test=load_data(argv[2],&nb_variable_test,&nb_exemple_test,&nb_classe_test);
  /*  float h=atof (argv[3]);
    //printf ("%d %d %d %d\n",nb_exemple_app,nb_exemple_test,nb_variable_app,nb_classe_app);
    int i=0;
    //for (i=0;i<nb_exemple_test;i++)
    //{
     //   printf ("%f\n",base_test[i][nb_variable_test]);
   // }
   int *tab_classe_mesure=parzen_classification(base_app,base_test,nb_variable_app,nb_classe_app,nb_exemple_test,nb_exemple_app,h);
    int taux_succes=0;
    for (i=0;i<nb_exemple_test;i++)
    {
        if (base_test[i][nb_variable_test]==tab_classe_mesure[i])
        {
            taux_succes+=1;
        }
       // printf ("%f %d\n",base_test[i][nb_variable_test],tab_classe_mesure[i]);
    }

    printf (" %f %f\n",h,(float)taux_succes/(float)nb_exemple_test);

    //printf("Hello world!\n");*/

  //  write_data_lib_svm (base_app, "train_libsvm",nb_variable_app , nb_exemple_app);
  //  write_data_lib_svm (base_test, "test_libsvm",nb_variable_test , nb_exemple_test);
    return 0;
}
