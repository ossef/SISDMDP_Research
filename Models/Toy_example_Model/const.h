/*constantes a modifier pour chaque modele*/

#define NEt               4   /*nombre de composantes du vecteur d'etat : (M, D, H, X)*/
#define NbArrivals        2   /*nombre d'arrivée, arrivées Batch {0,1,2} */
#define NbService         2   /*Service Bernouilli {0,1} */
#define NbRelease         2   /*Battery release {0,1} = Number of % release, example: All (X), Half (X/2) */
#define NbPhaseMeteo      2   /*Number of phases de Meteo = {0, 1} with two phases */
#define NbEvtsPossibles   NbArrivals*NbRelease*NbService*NbPhaseMeteo  /*nombre de vecteurs d'arrivees possibles*/
#define Polynom			0   /* iteration du modele */
#define Epsilon1 0.0		
#define Epsilon2  0.0	
	
