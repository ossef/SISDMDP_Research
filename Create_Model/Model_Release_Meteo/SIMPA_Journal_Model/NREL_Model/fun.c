
/*---------------------------------------------*/
/*                                             */
/*partie de Code specifique a  chaque probleme */
/*                                             */
/*---------------------------------------------*/

/* Modéle SISDMDP de remplissage de batterie avec des "Energy Packets" */	
/* DTMC Descitpion : (TypeMeteo, TypeDebut, Horloge, Batteri) */
/* DTMC Descitpion : (TM, TD, H, X) 
   Synthetic realistic model : 
   TM in {0, 1, 2, 3}            : Four Meteo Phases
   TD in {0, 1, 2  ... Buffer/2} : sell all X, sell X/2, sell X/4
   H in {0, ..., Deadline}
   X in {0, ..., Buffer}

   //Superstate 1 : (TM = 0, TD = 0, H, X)
   //Superstate 2 : (TM = 0, TD = 1, H, X)
   //Superstate 3 : (TM = 0, TD = 2, H, X)
        .
        .
        .
   //Superstate 4x(Buffer/2) : (TM = 0, TD = 0, H, X)

*/
		
/* Encoding discret events */
/* Arrivals = {0,1,2,3,4}, Service={0,1}, Release={0,1} */
/* index : (Arrival, Service, Release, Modulating) */
/* 1 : (0, 0, 0, 0) - no arrival, no service, no release, no phase change */
/* 2 : (0, 0, 0, 1) - no arrival, no service, no release, phase change */
/* 3 : (0, 0, 1, 0) - ...      ...      ...        ...       ...        */
/* 4 : (0, 0, 1, 1) - ...      ...      ...        ...       ...        */

/* 23 : (2, 1, 1, 0) - 2 arrivals, service, release, no phase change */
/* 24 : (2, 1, 1, 1) - 2 arrivals, service, release, phase change    */

/* proba des evenements : arrival, service, Release, MeteoChange */ 
extern double Action[NbRelease];    // stores vector of probabilities of current Action
extern int Buffer,  NbEvtsPossibles;

int b[NbService] = {0, 1};           // No service, one Service
int c[NbRelease] = {0, 1, 2, 3};     // Relase all battery, Release X, 2X/3, X/3, 0
int d[NbPhaseMeteo] = {0, 1, 2, 3};  // Two Meteo phases

double ps[2];                       //pService and pRelease for current action

/* An event structure: at each time-slot */
typedef struct event
{
  int a; //arrival
  int b; //service
  int c; //Battery release
  int d; //Meteophase change
  double prob; //Product of events probas
} Event;
Event *events; 

/* Empirical Arrivals distributions: P(A=a | M=m, hour=h_global) */
long double ***weather_hour_packet;   // [NbPhaseMeteo][num_hours_global][num_packets_global]

/* Empirical Service demand distribution per hour (aligned on global hour window) */
double *pService;                    // [num_hours_global]

/* Empirical Regime change matrix*/
long double Meteo[NbPhaseMeteo][NbPhaseMeteo];


/* Supports / sizes */
int a[100];                          // support values 0..a_size-1 (if needed)
int *a_size;                         // [1] = num_packets_global
int *num_packets;                    // [1] = num_packets_global
int *num_hours;                      // [1] = num_hours_global
int *Deadline;                       // [1] = global_end_hour

/* Per-regime metadata (requested) */
int start_hour_regime[NbPhaseMeteo];   // d_m
int end_hour_regime[NbPhaseMeteo];     // f_m
int num_packets_regime[NbPhaseMeteo];  // nb_paquets_m (=Amax_m+1)

/* Global hour window */
int global_start_hour;               // d_min across regimes
int global_end_hour;                 // f_max across regimes

/* Packet size (should be the same in all regime files) */
int packet_size_wh;                  // e.g., 200, 300, ...



void InitEvents(){
    int i, j, k, l;
    int compt = 0;

    int NbArrivals = a_size[0];
    printf("NbArrivals = %d \n",NbArrivals);
    events = malloc( (NbArrivals * NbService * NbRelease * NbPhaseMeteo) * sizeof(Event));

   for (i = 0; i < NbArrivals; i++) {
      for (j = 0; j < NbService; j++) {
        for (k = 0; k < NbRelease; k++) {
          for (l = 0; l < NbPhaseMeteo; l++) {
              events[compt].a = a[i]; 
              events[compt].b = b[j]; 
              events[compt].c = c[k];
              events[compt].d = d[l];
              events[compt].prob = Action[k]; //(double)pa[i]*ps[j]*Action[k];
              compt++;
              }
            }
          }
        }
}


void InitEtendue(int global_start_hour, int global_end_hour, int Buffer)
{
  Min[0] = 0;
  Max[0] = NbPhaseMeteo-1;
  Min[1] = 0;
  Max[1] = Buffer; //Buffer - (int)(Buffer/4); //max of remaining DP : max(B-B, B-B/2, B-B/4) (sell all, half, quarter)
  Min[2] = global_start_hour;
  Max[2] = global_end_hour;
  Min[3] = 0;
  Max[3] = Buffer;

 InitEvents();
}

/* =========================
   Helpers
   ========================= */
int min(int a, int b){
  return a<b ? a : b ;
}

int max(int a, int b){
  return a<b ? b : a ;
}

static void die(const char *msg) {
    printf("%s\n", msg);
    exit(0);
}

static void skip_until_hour_header(FILE *file) {
    char line[1000];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'H') return;   // "Heure\t..."
    }
    die("Erreur: header 'Heure' introuvable dans le fichier .data");
}

static void read_one_regime_file(
    const char *filepath,
    int *d, int *f, int *np, int *packet_size,
    int *hours_out,           // length (f-d+1)
    long double **mat_out     // [H][np], allocated by caller
) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        printf("Impossible d'ouvrir: %s\n", filepath);
        exit(0);
    }

    char line[1000];

    /* header text line */
    if (!fgets(line, sizeof(line), file)) { fclose(file);
                die("Erreur lecture header .data"); }

    /* line with d f np packet_size */
    if (fscanf(file, "%d %d %d %d", d, f, np, packet_size) != 4) { fclose(file);
                die("Erreur lecture dimensions (d f np packet_size) dans .data"); }

    int H = (*f) - (*d) + 1;

    /* skip to "Heure ..." */
    skip_until_hour_header(file);

    /* read matrix */
    for (int i = 0; i < H; i++) {
        if (fscanf(file, "%d", &hours_out[i]) != 1) {
            fclose(file);
            die("Erreur lecture heure");
        }
        for (int j = 0; j < *np; j++) {
            if (fscanf(file, "%Lf", &mat_out[i][j]) != 1) {
                fclose(file);
                die("Erreur lecture proba matrice");
            }
        }
    }
    fclose(file);
}

/* =================================================
   Main reader of Empirical Distribution (NREL Data)
   ================================================ */

int ReadDistribs(int Buffer, char city[100])
{

    a_size      = malloc(sizeof(int));
    num_packets = malloc(sizeof(int));
    num_hours   = malloc(sizeof(int));
    Deadline    = malloc(sizeof(int));

    /* ---------- PASS 1: read only metadata from each regime file ---------- */
    global_start_hour =  999999;
    global_end_hour   = -999999;
    int np_max = 0;
    packet_size_wh = -1;

    for (int m = 0; m < NbPhaseMeteo; m++) {
        char path[256];
        snprintf(path, sizeof(path), "../NREL_Extracts/%s/%s_M%d.data", city, city, m);

        FILE *file = fopen(path, "r");
        if (!file) { printf("Impossible d'ouvrir: %s\n", path);
                     exit(0);  }

        char line[1000];
        if (!fgets(line, sizeof(line), file)) { fclose(file);
                      die("Erreur lecture header .data (pass 1)"); }

        int d, f, np, psize;
        if (fscanf(file, "%d %d %d %d", &d, &f, &np, &psize) != 4) { fclose(file);
                    printf("Erreur lecture (d f np packet_size) dans %s\n", path); exit(0); }
        fclose(file);

        start_hour_regime[m]  = d;
        end_hour_regime[m]    = f;
        num_packets_regime[m] = np;

        if (d < global_start_hour) global_start_hour = d;
        if (f > global_end_hour)   global_end_hour   = f;
        if (np > np_max)           np_max            = np;

        if (packet_size_wh < 0) packet_size_wh = psize;
        else if (psize != packet_size_wh) {  printf("[ERROR] packet_size differs across regimes for %s:\n", city);
                                             exit(0); }
    }

    int H_global = global_end_hour - global_start_hour + 1;

    num_hours[0]   = H_global;
    Deadline[0]    = global_end_hour;
    num_packets[0] = np_max;
    a_size[0]      = np_max;

    /* fill support if needed */
    for (int k = 0; k < np_max && k < 100; k++) a[k] = k;

    printf("---> City %s\n", city);
    for (int m = 0; m < NbPhaseMeteo; m++) {
        printf("     Regime M%d: hours [%d..%d], packets [0..%d]\n", m, start_hour_regime[m], end_hour_regime[m], num_packets_regime[m]-1);
    }
    printf("---> Global hour window [%d..%d] (H=%d)\n", global_start_hour, global_end_hour, H_global);
    printf("---> Global packets support [0..%d] (packet_size=%d Wh)\n\n", np_max-1, packet_size_wh);

    /* ---------- Allocate 3D array weather_hour_packet ---------- */
    weather_hour_packet = malloc(NbPhaseMeteo * sizeof(long double **));
    for (int m = 0; m < NbPhaseMeteo; m++) {
        weather_hour_packet[m] = malloc(H_global * sizeof(long double *));
        for (int i = 0; i < H_global; i++) {
            weather_hour_packet[m][i] = malloc(np_max * sizeof(long double));
            /* default: no energy for hours outside regime window => P(A=0)=1 */
            weather_hour_packet[m][i][0] = 1.0L;
            for (int j = 1; j < np_max; j++) weather_hour_packet[m][i][j] = 0.0L;
        }
    }

    /* ---------- PASS 2: read full matrices for each regime and copy into global grid ---------- */
    for (int m = 0; m < NbPhaseMeteo; m++) {
        char path[256];
        snprintf(path, sizeof(path), "../NREL_Extracts/%s/%s_M%d.data", city, city, m);

        int d, f, np, psize;
        int Hm = end_hour_regime[m] - start_hour_regime[m] + 1;

        int *hours_tmp = malloc(Hm * sizeof(int));
        long double **mat_tmp = malloc(Hm * sizeof(long double *));
        for (int i = 0; i < Hm; i++) mat_tmp[i] = malloc(num_packets_regime[m] * sizeof(long double));

        read_one_regime_file(path, &d, &f, &np, &psize, hours_tmp, mat_tmp);

        /* copy into global arrays */
        for (int i = 0; i < Hm; i++) {
            int hour = hours_tmp[i];
            int gi = hour - global_start_hour;  // global hour index

            if (gi < 0 || gi >= H_global) continue;

            /* overwrite default with data */
            for (int j = 0; j < np; j++) {
                weather_hour_packet[m][gi][j] = mat_tmp[i][j];
            }
            /* if np < np_max, remaining probs stay at 0 (already initialized) */
        }

        for (int i = 0; i < Hm; i++) free(mat_tmp[i]);
        free(mat_tmp);
        free(hours_tmp);
    }

    /* ---------- Read Service Demand (same window as global) ---------- */

      const char *filename = "../NREL_Extracts/Service_Demand.data";
      FILE *service_file = fopen(filename, "r");
      if (!service_file) {
          printf("Erreur: Impossible d'ouvrir le fichier '%s'.\n", filename);
          exit(0);
      }

      pService = malloc(H_global * sizeof(double));
      for (int i = 0; i < H_global; i++) pService[i] = 0.0;

      int hour;
      double probability;
      while (fscanf(service_file, "%d %lf", &hour, &probability) == 2) {
          if (hour >= global_start_hour && hour <= global_end_hour) {
              pService[hour - global_start_hour] = probability;
          }
      }
      fclose(service_file);

    /* ------------- Reading Weather Regime Transition Matrix ------------- */

    char meteo_file[300];
    snprintf(
        meteo_file,
        sizeof(meteo_file),
        "../NREL_Extracts/%s/%s_Day_Change.data",
        city,
        city
    );

    FILE *fileM = fopen(meteo_file, "r");
    if (fileM == NULL) { printf("Erreur: Impossible d'ouvrir le fichier météo '%s'\n", meteo_file);
                        exit(0); }

    int nb_regimes;
    fscanf(fileM, "%d", &nb_regimes);

    if (nb_regimes != NbPhaseMeteo) {  printf("Erreur: NbPhaseMeteo attendu = %d, trouvé = %d\n", NbPhaseMeteo, nb_regimes);
                                        exit(0); }

    for (int i = 0; i < NbPhaseMeteo; i++) {
        for (int j = 0; j < NbPhaseMeteo; j++) {
            fscanf(fileM, "%Lf", &Meteo[i][j]);
        }
    }

    fclose(fileM);

    // Debug (optional)
    printf("Matrice de transition météo :\n");
    for (int i = 0; i < NbPhaseMeteo; i++) {
        for (int j = 0; j < NbPhaseMeteo; j++)
            printf("%Lf ", Meteo[i][j]);
        printf("\n");
    }
    

    /* ---------- Your existing init ---------- */
    InitEtendue(global_start_hour, global_end_hour, Buffer);

    /* number of possible events per time step */
    return (a_size[0] * NbService * NbRelease * NbPhaseMeteo);
}


void EtatInitial(E)
int *E;
{
  /*donne a E la valeur de l'etat racine de l'arbre de generation*/
  E[0] = Min[0];  //TypeMeteo : 0
  E[1] = Min[1];  //TypeDebut : 0
  E[2] = start_hour_regime[E[0]];  //Horloge   : 6h
  E[3] = Min[3];  //Batterie  : 0
}


double Probabilite(indexevt, E)
int indexevt;
int *E;
{

  Event e = events[indexevt-1];
  double prob;

  //--- 1) The Release: already calculated in InitEvents()
  double p = events[indexevt-1].prob;

  //--- 2) Meteo change
  prob = Meteo[E[0]][e.d] * p;

  //---  3) Service of a DP (Data Packet)
  if(e.b == 1)
    prob *= pService[E[2]-Min[2]];
  else 
    prob *= (1-pService[E[2]-Min[2]]);

  //--- 4) The arrival of EP (Energy Packet)
  prob *= weather_hour_packet[E[0]][E[2]-Min[2]][e.a];

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
  int d[NbPhaseMeteo] = {0, 1, 2, 3}; 
  */

  // start and end hour depends on the regime
  int minH = start_hour_regime[E[0]];
  int maxH = end_hour_regime[E[0]];

  // -------- X --------- : Soc Update
  if(E[2] < maxH) 
      F[3] = min(Max[3], max(0, E[3]+e.a - e.b));
  if(E[2] == maxH)  //|| (E[2] < maxH && E[3] == Max[3]) )
    {
      if(e.c == 0)  //Sell all battery
        F[3] = 0; 
      if(e.c == 1)  //Sell 2/3 of battery
        F[3] = E[3] - (2*E[3])/3;
      if(e.c == 2)  //Sell 1/3 of battery
        F[3] = E[3] - E[3]/3;
      if(e.c == 3)  //Don't sell battery
        F[3] = E[3];
    }

  // -------- H --------- : Time Update
  if(E[2] < maxH)
      F[2]++;
  if(E[2] == maxH)
      F[2] = minH;

  // -------- Td --------- : TypeDebut Update
  if(E[2] == maxH)  //|| (E[2] < maxH && E[3] == Max[3]) )
    {
      if(e.c == 0)  //Sell all battery
        F[1] = 0; 
      if(e.c == 1)  //Sell 2/3 of battery
        F[1] = E[3] - (2*E[3])/3;
      if(e.c == 2)  //Sell 1/3 of battery
        F[1] = E[3] - E[3]/3;
      if(e.c == 3)  //Don't sell battery
        F[1] = E[3];
    }

  // -------- Tm --------- : TypeMeteo Update
  if(E[2] == maxH) // || (E[2] < maxH && E[3] == Max[3]) )
    F[0] = e.d;
  
  //printf("\n j = (%d, %d, %d, %d)",e.a,e.b,e.c,e.d);

  //if(E[0] == 0)
  //  printf("Transition (%d, %d, %d, %d) --> (%d, %d, %d, %d) : (%d, %d, %d, %d) \n",E[0],E[1],E[2],E[3], F[0],F[1],F[2],F[3],e.a,e.b,e.c,e.d);
}
