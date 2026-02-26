/*constantes a modifier pour chaque modele*/

#define NEt               4   /*nombre de composantes du vecteur d'etat : (M, D, H, X)*/
//#define NbArrivals      5   /*nombre d'arrivée, arrivées Batch {0,1,2,3,4}. Here will depend on NREL data */
#define NbService         2   /*Service Bernouilli {0,1} */
#define NbRelease         4   /*Battery release {0,1,2,3} = Number of % release, example: sell (X), sell (2X/3), sell (X/3), sell 0  */
#define NbPhaseMeteo      4   /*Number of phases de Meteo = {0,1,2,3} with four phases */
#define Polynom			  0   /* iteration du modele */
#define Epsilon1 0.0		
#define Epsilon2  0.0	
	
