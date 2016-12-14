/****************************************************************/
/*	Reseau multicouche, avec algorithme d'apprentissage BP      */
/*	------------------------------------------------------	    */
/*		Helene Paugam-Moisy	-	17/03/16                        */
/* VERSION multiclasses, en langage C - TD L3 Sc Co - Info-CNX  */
/****************************************************************/

//inclusion des bibliotheques d'utilitaires

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////

//debut des structures de donnees

#define coef 2.0   //pente de la fonction sigmoide en zero
#define coed 1.0   //demi-pente (pour l'oprerateur derive)

// sera lue dans le fichier #define diment 26  //dimension de l'espace d'entree
// MODIFIC 1 : sera lue dans le fichier #define dimsor 1   //dimension de l'espace de sortie = nb de classes
#define maxcar 5000  //nombre maximum de caracteres pour les noms des fichiers

typedef struct // -- structure d'un neurone --
{
    int nbent;	//nombre d'entrees du neurone
    double *x;	//tableau des entrees X
    double *w;	//tableau des poids W
    double a;	//activation (somme ponderee des entrees = somme(WiXi))
    double s;	//sortie (fonction sigmoide de l'activation =f(a))
    double y;	//gradient
}strucNeurone;


typedef struct // -- structure d'une couche --
{
    int nbneu;		//nombre de neurones
    strucNeurone *neurone;	//tableau de neurones
}strucCouche;


typedef struct // -- structure d'un reseau --
{
  int nbcou;		//nombre de couches
  strucCouche *couche;	//tableau de couches
}strucReseau;

typedef struct // -- base d'exemples --
{
  int nbex;     //nombre d'exemples <= sera lu dans le fichier
  double **xx;  //vecteurs des entrees
  double **sd;  //vecteurs des sorties desirees
}strucBase;

strucReseau res;  //le reseau MLP

char nficapp[maxcar]; //nom du fichier d'apprentissage
char nficgen[maxcar]; //nom du fichier de generalisation
strucBase app,gen;    //les bases d'apprentissage et de generalisation

char nresult[maxcar]; //nom du fichier de stockage des resultats
FILE *ficresu;        //fichier dans lequel on stockera les parametres et les resultats

float alpha;  //coefficient d'apprentissage (pas de gradient)
int nbpass;   //nombre de passes de la base d'exemples
int diment;   //dimension d'entree <= sera lue dans le fichier
int dimsor;   // MODIFIC 1 (suite et fin) : dimension de sortie <= sera lue dans le fichier
//int nbcou;

//fin des structures de donnees

/////////////////////////////////////////////////////////////////

//fonctions mathematiques ///////////////////

double sigmo(double a)    //fonction sigmoide -- s'applique a l'activation a
{
	double aux;

	aux=exp(coef*a);
	return((aux-1)/(aux+1));  //fonction impaire, a valeurs dans ]-1;+1[
}

double deriv(double fa)   //operateur derive -- s'applique a la sortie s=f(a)
{
  return(coed*(1+fa)*(1-fa));  //formule pour calculer f'(a) a partir de f(a)
}

//procedures d'initialisation ///////////////////

void initNeurone(int i, int k)
{
  int j;

  // printf("initialisation du neurone %d de la couche %d\n",i+1,k);
  res.couche[k].neurone[i].nbent=res.couche[k-1].nbneu;
  res.couche[k].neurone[i].x=(double*)malloc(sizeof(double)*res.couche[k].neurone[i].nbent);
  res.couche[k].neurone[i].w=(double*)malloc(sizeof(double)*res.couche[k].neurone[i].nbent);;
  for (j=0;j<res.couche[k].neurone[i].nbent;j++)
    res.couche[k].neurone[i].w[j]=0.6*rand()/RAND_MAX-0.3;  //poids initiaux dans ]-0.3;+0.3[
}

void initCouche(int k,char *argv7)
{
  int i;

  //printf("initialisation de la couche %d\n",k);
  if (k==0)                 // couche d'entree
    res.couche[k].nbneu=diment;
  else if (k==res.nbcou-1)  // couche de sortie
    res.couche[k].nbneu=dimsor;
  else                      // couches cachees
    { //printf("\nNombre de neurones sur la couche cachee %d : ",k);
      //scanf("\n%d",&res.couche[k].nbneu);
      res.couche[k].nbneu=atoi(argv7);
      fprintf(ficresu,"Nombre de neurones sur la couche cachee %d : %d\n",k,res.couche[k].nbneu);
    }
  res.couche[k].neurone=(strucNeurone*)malloc(sizeof(strucNeurone)*res.couche[k].nbneu);
  if (k>0)
    for (i=0;i<res.couche[k].nbneu;i++)  //initialisation de chaque neurone i de la couche k
        initNeurone(i,k);
}

void initReseau(char *argv6,char *argv7)
{
  int k,nbcach;

 // printf("initialisation du reseau MLP\n");
  //printf("\nNombre de couches cachees : ");
  //scanf("\n%d",&nbcach);
  nbcach=atoi(argv6);
  res.nbcou=nbcach+2;
  fprintf(ficresu,"Nombre de couches cachees : %d\n",nbcach);

  res.couche=(strucCouche*)malloc(sizeof(strucCouche)*(nbcach+2));
  for (k=0;k<res.nbcou;k++)
    { //printf("\n -- appel initCouche avec k=%d\n",k);
    initCouche(k,argv7);}
}

void initBase(char *nomfich, strucBase *base)
{
  int i,j;
  char c;
  FILE *fich;

 // printf("Lecture des motifs, a partir du fichier %s\n",nomfich);
  fich=fopen(nomfich,"r");
  fscanf(fich,"%d",&(base->nbex));     //le nombre d'exemples est en 1ere ligne du fichier
//  printf("nbex = %d \n",base->nbex);
  fscanf(fich,"%d",&diment);   //la dim. des entrees est en 2eme ligne du fichier
  //printf("diment = %d \n",diment);
  // MODIFIC 2 : lecture du nb de sorties a partir du fichier de donnees
  fscanf(fich,"%d",&dimsor);   //la dim. des sorties est en 3eme ligne du fichier
  //printf("dimsor = %d \n",dimsor);
  // fin de MODIFIC 2
  base->xx=(double**)malloc(sizeof(double)*base->nbex);
  base->sd=(double**)malloc(sizeof(double)*base->nbex);
  for (i=0;i<base->nbex;i++)         //on lit, l'un apres l'autre, tous les exemples de la base
    { base->xx[i]=(double*)malloc(sizeof(double)*diment); //le vecteur des entrees, suivi de...
      for (j=0;j<diment;j++)
      	fscanf(fich,"%lf",&(base->xx[i][j]));
      base->sd[i]=(double*)malloc(sizeof(double)*dimsor); //...le vecteur des sorties desirees
      for (j=0;j<dimsor;j++)
	    fscanf(fich,"%lf",&(base->sd[i][j]));
    }
  fclose(fich);
  //printf("fin de la lecture des motifs\n");
}

//procedures de presentation d'un exemple au reseau ///////////////////

void prezEntrees(int numex, strucBase base)  //presentation d'un exemple au reseau
{
  int i;

//les composantes de l'exemple numex sont les sorties des neurones de la couche 0 du reseau
  for (i=0;i<diment;i++)
  {
      res.couche[0].neurone[i].s=base.xx[numex][i];
//      printf(" xx[%d,%d]=%lf",numex,i,base.xx[numex][i]);
  }
}

//procedures de fonctionnement du reseau ///////////////////

void aller()  //passe avant : calcul de la sortie du reseau, pour l'exemple presente
{
  int i,j,k;

  for (k=1;k<res.nbcou;k++)
    for (i=0;i<res.couche[k].nbneu;i++)
      { res.couche[k].neurone[i].a=0.0;
	for (j=0;j<res.couche[k].neurone[i].nbent;j++)
	  { res.couche[k].neurone[i].x[j]=res.couche[k-1].neurone[j].s;
	    res.couche[k].neurone[i].a+=res.couche[k].neurone[i].w[j]*res.couche[k].neurone[i].x[j];
	  }
	res.couche[k].neurone[i].s=sigmo(res.couche[k].neurone[i].a);
      }
}

void gradsor(int numex)  //calcul des gradients d'erreur, sur la couche de sortie
{
  int i;
  double dsig;  //derivee de la sigmoide

  for (i=0;i<dimsor;i++)
    { dsig=deriv(res.couche[res.nbcou-1].neurone[i].s);
      res.couche[res.nbcou-1].neurone[i].y=2*dsig*(app.sd[numex][i]-res.couche[res.nbcou-1].neurone[i].s);
    }
}

void retour()  //passe arriere : calcul des gradients retropropages, dans les couches cachees
{
  int i,m,k;
  double dsig;
  double somm;

  for (k=res.nbcou-2;k>0;k--)
    for (i=0;i<res.couche[k].nbneu;i++)
      { dsig=deriv(res.couche[k].neurone[i].s);
	somm=0.0;
	for(m=0;m<res.couche[k+1].nbneu;m++)
	  somm+=res.couche[k+1].neurone[m].w[i]*res.couche[k+1].neurone[m].y;
	res.couche[k].neurone[i].y=dsig*somm;
      }
}

void modifw()  //mise a jour des poids du reseau (toutes les couches)
{
  int i,j,k;

  for (k=1;k<res.nbcou;k++)
    for (i=0;i<res.couche[k].nbneu;i++)
      for (j=0;j<res.couche[k].neurone[i].nbent;j++)
	res.couche[k].neurone[i].w[j]+=alpha*res.couche[k].neurone[i].y*res.couche[k].neurone[i].x[j];
}

//fonctions d'evaluation des perfromances ///////////////////

//int tstsor(int numex, strucBase base)  //resultat booleen -- version pour dimsor=1
//{
////test actuel implemente pour le seul cas ou il n'y a qu'une seule sortie, valant -1 ou +1
//    if (res.couche[res.nbcou-1].neurone[0].s*base.sd[numex][0]>0)
//      return(1);
//    else
//      return(0);
//}

// MODIFIC 3 : re-ecriture de la fonction tstsor()
int tstsor(int numex, strucBase base)    // version multiclasse, pour dimsor = nb de classes
{
    double maxi=-1;
    int i,numneur=0;

    for (i=0;i<dimsor;i++)
        if (res.couche[res.nbcou-1].neurone[i].s>maxi)
        {
            maxi=res.couche[res.nbcou-1].neurone[i].s;
            numneur=i;
        }
    if (base.sd[numex][numneur]==1)
        { //printf(" BON ");
            return(1); }
    else
        { //printf(" mal ");
            return(0); }
}
// fin de la MODIFIC 3

double tauxSucces(strucBase base)
{
  int i,nbsuc;

  nbsuc=0;
  for(i=0;i<base.nbex;i++)
    { prezEntrees(i,base);
      aller();
      if (tstsor(i,base)==1)
	nbsuc++;
    }
  return((double) nbsuc/base.nbex);
}

//procedure de controle des processus d'apprentissage et de generalisation ///////////////////

void appgen(int nbpass) //apprentissage avec test en generalisation a chaque passe
{
  int p,i;
  double taux;

  for (p=0;p<nbpass;p++)
    { //printf("%d\t",p+1);
     // fprintf(ficresu,"Passe n. %d\t",p+1);
      for (i=0;i<app.nbex;i++) //une passe d'apprentissage sur toute la base d'exemples
	{ // printf("\n Exemple %d : ",i);
	  prezEntrees(i,app);
	  aller();
	  gradsor(i);
	  retour();
	  modifw();
	}
      taux=tauxSucces(app); //evaluation du taux de succes en apprentissage
      //printf("taux app : %lf\t",taux);
      //fprintf(ficresu,"taux app : %lf\t",taux);
      taux=tauxSucces(gen); //evaluation du taux de succes en generalisation
    if (p==999)
    { printf(" %lf\n",taux);
    }
      fprintf(ficresu," %lf\n",taux);
    }
}

/////////////////////////////////////////////////////////////////

//programme principal ///////////////////

void main(int argc , char ** argv)
{
  srand(1234);  // graine ("seed") initiale du generateur de nombres aleatoires
  double tt1,tt2;

//Nom du fichier d'apprentissage
strcpy (nficapp,argv[1]);
  //Nom du fichier de generalisation

  strcpy (nficgen,argv[2]);
  //Nom du fichier de resultats

  strcpy (nresult,argv[3]);

  ficresu=fopen(nresult,"w");


  //fprintf(ficresu,"Apprentissage sur %s\t",nficapp);
  // printf ("338\n");
  //printf("\n -- appel initBase, pour les donnees d'apprentissage\n");
  initBase(nficapp,&app);
  fprintf(ficresu,"  [ %d exemples ]\n",app.nbex);
  fprintf(ficresu,"Generalisation sur %s\t",nficgen);
  //printf("\n -- appel initBase, pour les donnees de generalisation\n");
  initBase(nficgen,&gen);
  fprintf(ficresu,"  [ %d exemples ]\n",gen.nbex);

 //Valeur du pas de gradient
 alpha=atof(argv[4]);
 //printf ("341\n");
  fprintf(ficresu,"alpha=%f\t",alpha);
//Nombre de passes de la base d'exemples
  nbpass=atoi(argv[5]);
//printf ("344\n");
  fprintf(ficresu,"nbpass=%d\n",nbpass);

  fprintf(ficresu,"Taille de la couche d'entree = %d\n",diment);
  fprintf(ficresu,"Taille de la couche de sortie = %d\n",dimsor);
 // printf("\n -- appel initReseau\n");
  initReseau(argv[6],argv[7]);

 // printf("\n\nPhase d'apprentissage, avec test en apprentissage et en generalisation, apres chaque passe de la base d'exemples.\n\n");
tt1=clock();
  appgen(nbpass);
tt2=clock(); //printf("temps--->%f s\n",(tt2-tt1)*0.000001);
  fprintf(ficresu,"\ntemps--->%f s\n",(tt2-tt1)*0.000001);

  fclose(ficresu);
}

//fin du programme

/////////////////////////////////////////////////////////////////
