
/*---------------------------------------------*/
/*                                             */
/*partie de Code specifique a  chaque probleme */
/*                                             */
/*---------------------------------------------*/

/* Modéle SISDMDP de remplissage de batterie avec des "Energy Packets" */	
/* DTMC Descitpion : (TypeMeteo, TypeDebut, Horloge, Batteri) */
/* DTMC Descitpion : (TM, TD, H, X) 
   Toy example : 
   TM in {0, 1} : two Meteo Phases
   TD in {0, 1} : sell all, sell 1/2 
   X in {0, ..., 5}
   H in {0, ..., 3}

*/
		
/* Encoding discret events */
/* Arrivals = {0,1,2}, Service={0,1}, Release={0,1} */
/* index : (Arrival, Service, Release, Modulating) */
/* 1 : (0, 0, 0, 0) - no arrival, no service, no release, no phase change */
/* 2 : (0, 0, 0, 1) - no arrival, no service, no release, phase change */
/* 3 : (0, 0, 1, 0) - ...      ...      ...        ...       ...        */
/* 4 : (0, 0, 1, 1) - ...      ...      ...        ...       ...        */

/* 23 : (2, 1, 1, 0) - 2 arrivals, service, release, no phase change */
/* 24 : (2, 1, 1, 1) - 2 arrivals, service, release, phase change    */


/* proba des evenements : arrival, service, Release, MeteoChange */ 

extern double Action[NbRelease];
extern int Deadline, Buffer;

int a[NbArrivals] = {0, 1};  // 0, 1 or 2 arrivals
int b[NbService] = {0, 1};    // No service, one Service
int c[NbRelease] = {0, 1};    // Relase all Battery, Release Half of the battery
int d[NbPhaseMeteo] = {0, 1}; // Two Meteo phases
float Meteo[NbPhaseMeteo][NbPhaseMeteo] = {  
    {0.3, 0.7},  // M1 -> M2 : 0.7
    {0.7, 0.3}   // M2 -> M1 : 0.7
};

//Not depending on action
double pa[NbArrivals] = {0.2, 0.8};    //proba of arrivals, avg 1.3 arrival

double pService[NbRelease] = {0.1, 0.1};       //Bernouilli service, for each action
//double pRelease[NbRelease] = {0.6, 0.4};       //Bernouilli release of Battery, for each action

//pService and pRelease for current action
double ps[2];

int min(int a, int b){
  return a<b ? a : b ;
}

int max(int a, int b){
  return a<b ? b : a ;
}

typedef struct event
{
  int a; //arrival
  int b; //service
  int c; //Battery release
  int d; //Meteophase change
  double prob; //Product of events probas
} Event;
Event events[NbEvtsPossibles];

void InitEvents(){
   int i, j, k, l;
   int compt = 0;

   ps[1] = pService[0]; ps[0] = 1 - ps[1]; // Service probability
   for (i = 0; i < NbArrivals; i++) {
      for (j = 0; j < NbService; j++) {
        for (k = 0; k < NbRelease; k++) {
          for (l = 0; l < NbPhaseMeteo; l++) {
              events[compt].a = a[i]; 
              events[compt].b = b[j]; 
              events[compt].c = c[k];
              events[compt].d = d[l];
              events[compt].prob = (double)pa[i]*ps[j]*Action[k];
              compt++;
              }
            }
          }
        }
}

void InitEtendue()
{
  Min[0] = 0;
  Max[0] = 1;
  Min[1] = 0;
  Max[1] = (int)(Buffer/2); // 6 = 12/2
  Min[2] = 0;
  Max[2] = Deadline;
  Min[3] = 0;
  Max[3] = Buffer;

  InitEvents();
}


void EtatInitial(E)
int *E;
{
  /*donne a E la valeur de l'etat racine de l'arbre de generation*/
  E[0] = 0;  //TypeMeteo
  E[1] = 0;  //TypeDebut
  E[2] = 0;  //Horloge
  E[3] = 0;  //Batterie
}


double Probabilite(indexevt, E)
int indexevt;
int *E;
{
  Event e = events[indexevt-1];
  double p = events[indexevt-1].prob;
  double prob = 0;

  //--- Meteo change
  prob = Meteo[E[0]][e.d] * p;
  //prob = p;

  /*
  //--- Day 
  if(E[2] == 1)
  {
    if(e.d == 1)
      prob = pAlpha*p;
    else
      prob = (1-pAlpha)*p;
  }

  //--- Night
  if(E[2] == 0)
  {
    if(e.d == 1)
      prob = pBeta*p;
    else
      prob = (1-pBeta)*p;
  }
  */
  return prob;
}


void Equation(E, indexevt, F, R)
int *E;
int indexevt;
int *F, *R;
{
  /*ecriture de l'equation d'evolution, transformation de E en F grace a l'evenemnt indexevt, mesure de la recompense R sur cette transition*/
  int bool = 0;
  F[0] = E[0];  // TypeMeteo, Tm
  F[1] = E[1];  // TypeDebut, Td
  F[2] = E[2];  // Horloge, H
  F[3] = E[3];  // Batterie, X
  Event e = events[indexevt-1];

  /* 
  int a[NbArrivals] = {0, 1, 2, 3}; 
  int b[NbService] = {0, 1};
  int c[NbRelease] = {0, 1, 2};
  int d[NbPhaseMeteo] = {0, 1}; 
  */

  // -------- X --------- : Soc Update
  if(E[2] < Max[2]) 
      F[3] = min(Max[3], max(0, E[3]+e.a - e.b));
  if( E[2] == Max[2])
    {
      if(e.c == 0)  //Sell all battery
        F[3] = 0; 
      if(e.c == 1)  //Sell 1/2 of battery
        F[3] = (int) (E[3]/2);
      //if(e.c == 2)  //Sell 1/4 of battery
      //  F[3] = (int) (E[3]/4);
    }

  // -------- H --------- : Time Update
  if(E[2] < Max[2])
      F[2]++;
  if(E[2] == Max[2])
      F[2] = 0;

  // -------- Td --------- : TypeDebut Update
  if( E[2] == Max[2])
    {
      if(e.c == 0)  //Sell all battery
        F[1] = 0; 
      if(e.c == 1)  //Sell 1/2 of battery
        F[1] = (int) (E[3]/2); 
      //if(e.c == 2)  //Sell 1/4 of battery
      // F[1] = (int) (E[3]/4);
    }

  // -------- Tm --------- : TypeMeteo Update
  if( E[2] == Max[2])
    F[0] = e.d;

  //if(E[0] == 0)
  //  printf("Transition (%d, %d, %d, %d) --> (%d, %d, %d, %d) : (%d, %d, %d, %d) \n",E[0],E[1],E[2],E[3], F[0],F[1],F[2],F[3],e.a,e.b,e.c,e.d);
}

void InitParticuliere()
{}

