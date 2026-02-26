#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "const.h"
#define DEBUG 0
#define MAXETATS 10000

void afficherEtats(int t1[MAXETATS],int t2[MAXETATS],int t3[MAXETATS],int t4[MAXETATS],int t5[MAXETATS],int n)
{
	printf("[\n");
		for(int i=0;i<n;i++)
		printf("%d : %d  %d  %d %d \n",t1[i],t2[i],t3[i],t4[i],t5[i]);
	printf("] \n");
}

void TrierEtats_Cd(int t1[MAXETATS],int t2[MAXETATS],int t3[MAXETATS],int t4[MAXETATS],int t5[MAXETATS],int n)
{
    int i, j;
    for (i = 1; i < n; i++) {

        // Sauvegarde de la ligne i
        int i1 = t1[i];
        int i2 = t2[i];
        int i3 = t3[i];
        int i4 = t4[i];
        int i5 = t5[i];

        j = i - 1;

        // Comparaison lexicographique : M → D → H → X
        while (j >= 0 &&
              (t2[j] > i2 ||
              (t2[j] == i2 && t3[j] > i3) ||
              (t2[j] == i2 && t3[j] == i3 && t4[j] > i4) ||
              (t2[j] == i2 && t3[j] == i3 && t4[j] == i4 && t5[j] > i5)))
        {
            t1[j + 1] = t1[j];
            t2[j + 1] = t2[j];
            t3[j + 1] = t3[j];
            t4[j + 1] = t4[j];
            t5[j + 1] = t5[j];
            j--;
        }

        // Insertion
        t1[j + 1] = i1;
        t2[j + 1] = i2;
        t3[j + 1] = i3;
        t4[j + 1] = i4;
        t5[j + 1] = i5;
    }
}

void EcrirePartitions_MD_FILE(FILE *fpart,int M[MAXETATS],int D[MAXETATS],int H[MAXETATS],int X[MAXETATS],int n)
{
    if (n <= 0) {
        fprintf(fpart, "0\n");
        return;
    }

    /* Compter le nombre de partitions définies par (M,D) */
    int nb = 1;
    for (int i = 1; i < n; i++) {
        if (M[i] != M[i-1] || D[i] != D[i-1]) nb++;
    }

    fprintf(fpart, "%d\n", nb);

    /* Écrire chaque partition (blocs contigus après tri) */
    int part_id = 0;
    int start = 0;

    for (int i = 1; i <= n; i++) {
        int fin_partition = (i == n) || (M[i] != M[i-1]) || (D[i] != D[i-1]);

        if (fin_partition) {
            int end = i - 1;

            /* Format: id | first last | M D H X (du premier état) */
            fprintf(fpart, "%d | %d %d | %d %d %d %d\n",
                    part_id,
                    start, end,    // start/end = indices dans le tableau trié (nouvel index des états)
                    M[start], D[start], H[start], X[start]);

            part_id++;
            start = i;
        }
    }
}

int Rechercher(int I[MAXETATS],int e,int n)
{
for(int i=0;i<n;i++)
	 {
		if (I[i] == e)
		return i; 	 
     }		
	 return -1;
}

void OrdonnerMatrice_Rii(FILE *frii,int I[MAXETATS],char s[100],char rii[100],int n)
{
FILE *friiout;
char  riiout[100];
int i,j,c;
int t=0;

strcpy(riiout,s);
strcat(riiout,"-reordre.Rii");

friiout = fopen(riiout,"w");
if(friiout == NULL)
 exit(2);


int iter,degre,e,k,arret;
long double proba;
c= 0;

while(c <n)
 {
	for(i=0;i<n;i++)
	 {
		fscanf(frii,"%d %d",&iter,&degre);
		if(iter == I[c])
		 {
			  fprintf(friiout,"%12d %12d ",t++,degre);
			  for(j=0;j<degre;j++)
			  {
				fscanf(frii,"%Le %d",&proba,&e);
				fprintf(friiout,"% .15LE%12d",proba,Rechercher(I,e,n)); // recherche le nouvel emplacement de l'etats
			  }
				fprintf(friiout,"\n");
			c++;	
		 }
		 else
		 {
			   for(j=0;j<degre;j++)
				fscanf(frii,"%Le %d",&proba,&e);
		 }
	 }
	 frii=fopen(rii,"r");
 }
 
 fclose(friiout);	
}

int main(int argc, char *argv[])
{
 FILE *fcd,*fsz,*fpi,*friiout,*fszout,*fcdout,*frii, *fpart;
 char sz[100], cd[100], rii[100],szout[100], cdout[100], part[100];
 int I[MAXETATS];
 int M[MAXETATS];
 int D[MAXETATS];
 int H[MAXETATS];
 int X[MAXETATS];
 long double proba,seuil;
 int deadline;
 int i,n1,n2,n; 

 
 if(argc != 2)
 {
 	printf(" Erreur passez en parametre le nom du model \n <Exemple d'usage > ./reordonner model50 \n");
 	exit(1);
 }
	 
	 strcpy(sz,argv[1]); 
	 strcpy(cd,argv[1]);
	 strcpy(rii,argv[1]);
	 strcpy(szout,argv[1]); 
	 strcpy(cdout,argv[1]);
	 strcpy(part,argv[1]);


  	 strcat(sz,".sz"); // On concatène chaine2 dans chaine1
  	 strcat(cd,".cd");
  	 strcat(rii,".Rii");
  	 strcat(szout,"-reordre.sz"); 
	 strcat(cdout,"-reordre.cd");
	 strcat(part,"-reordre.part");


 fsz = fopen(sz,"r");
 if(fsz == NULL)
 exit(2);


 fcd = fopen(cd,"r");
 if(fcd == NULL)
 exit(2);


 frii = fopen(rii,"r");
 if(frii == NULL)
 exit(2);


 fszout = fopen(szout,"w");
 if(fszout == NULL)
 exit(2);

 
 fcdout = fopen(cdout,"w");
 if(fcdout == NULL)
 exit(2);


 fpart = fopen(part,"w");
 if(fpart == NULL)
 exit(2);


/*-------------------- Lecture du fichier ".sz" et ecriture dans ".reorder.sz" --------------*/

 fscanf(fsz,"%d",&n1);
 fscanf(fsz,"%d",&n);
 fscanf(fsz,"%d",&n2);
 fprintf(fszout,"%12d \n%12d \n%12d \n",n1,n,n2);
 
 
/*-------------------- Lecture du fichier ".cd" -----------------*/

for(i=0;i<n;i++)
 {
	fscanf(fcd,"%d %d %d %d %d",&I[i],&M[i],&D[i],&H[i],&X[i]);
 }


if(DEBUG == 1)
{
  printf("Avant trie :\n");
  afficherEtats(I,M,D,H,X,n);
}

/*------------------ Tri des états !! ----------------------------*/

TrierEtats_Cd(I,M,D,H,X,n);

/*------------------ Ecriture des partitions !! ----------------------------*/

EcrirePartitions_MD_FILE(fpart, M, D, H, X, n);


if(DEBUG == 1)
{  
  printf("\n Aprés trie :\n");
  afficherEtats(I,M,D,H,X,n);
}

/*-----------------Ecriture du nouveau fichier reorder.cd .........*/

  for(i=0;i<n;i++)
	fprintf(fcdout,"%12d %12d %12d %12d %12d\n",i,M[i],D[i],H[i],X[i]);
	

/*---------------- Re ordonner et ecriture de la nouvel matrice ".reorder.Rii" !! -----*/
 
  OrdonnerMatrice_Rii(frii,I,argv[1],rii,n); 

	fclose(fsz);
	fclose(fcd);
	fclose(frii);
	fclose(fszout);
	fclose(fcdout);
	fclose(fpart);
	printf("Matrix re-order done ! \n");
	
  return 0;
}
