/*
 * Re-order-Matrice-New.c  — Re-ordonner une chaîne de Markov sparse (format Rii)
 * Ordre lexicographique des états : M croissant, puis D, puis H, puis X.
 *
 * Entrée :
 *   <model>.sz   : nnz, nStates, dim
 *   <model>.cd   : old_id  M D H X
 *   <model>.Rii  : (par ligne) old_row  degre   (proba  old_col) * degre
 *   <model>.pi   : N lignes, pi[1]..pi[N] (ordre original, 1-based comme GTHCreux)
 *
 * Sortie :
 *   <model>-reordre.sz
 *   <model>-reordre.cd
 *   <model>-reordre.part
 *   <model>-reordre.Rii
 *   <model>-reordre.pi   (pi permuté dans le nouvel ordre)
 *
 * Compilation :
 *   gcc -O2 Re-order-Matrice-New.c -o ReOrder-New
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define MAXETATS 150000

typedef struct {
    int old_id;
    int M, D, H, X;
} State;

typedef struct {
    int col;
    long double val;
} Edge;

static void die(const char *msg) {
    perror(msg);
    exit(2);
}

static int cmp_state(const void *a, const void *b) {
    const State *sa = (const State*)a;
    const State *sb = (const State*)b;

    if (sa->M != sb->M) return (sa->M < sb->M) ? -1 : 1;
    if (sa->D != sb->D) return (sa->D < sb->D) ? -1 : 1;
    if (sa->H != sb->H) return (sa->H < sb->H) ? -1 : 1;
    if (sa->X != sb->X) return (sa->X < sb->X) ? -1 : 1;

    if (sa->old_id != sb->old_id) return (sa->old_id < sb->old_id) ? -1 : 1;
    return 0;
}

static int cmp_edge_col(const void *a, const void *b) {
    const Edge *ea = (const Edge*)a;
    const Edge *eb = (const Edge*)b;
    if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
    return 0;
}

/*
 * Écrit le fichier partitions MD :
 * nbPartitions
 * part_id | start end | M D H X (du premier état de la partition)
 */
static void EcrirePartitions_MD_FILE_State(FILE *fpart, const State *states, int n) {
    if (n <= 0) { fprintf(fpart, "0\n"); return; }

    int nb = 1;
    for (int i = 1; i < n; i++) {
        if (states[i].M != states[i-1].M || states[i].D != states[i-1].D) nb++;
    }

    fprintf(fpart, "%d\n", nb);

    int part_id = 0;
    int start = 0;

    for (int i = 1; i <= n; i++) {
        int fin = (i == n) ||
                  (states[i].M != states[i-1].M) ||
                  (states[i].D != states[i-1].D);

        if (fin) {
            int end = i - 1;
            fprintf(fpart, "%d | %d %d | %d %d %d %d\n",
                    part_id, start, end,
                    states[start].M, states[start].D,
                    states[start].H, states[start].X);
            part_id++;
            start = i;
        }
    }
}

/*
 * Lecture de <model>.Rii en CSR (rowptr/col/val).
 * Suppose une ligne par état (n lignes).
 */
static void read_Rii_to_CSR(FILE *frii, int n, int nnz,
                           int *rowptr, int *col, long double *val)
{
    int iter, degre;
    long double p;
    int e;

    rowptr[0] = 0;
    int pos = 0;

    for (int i = 0; i < n; i++) {
        if (fscanf(frii, "%d %d", &iter, &degre) != 2) {
            fprintf(stderr, "Erreur lecture Rii: header ligne %d\n", i);
            exit(2);
        }

        rowptr[i+1] = rowptr[i] + degre;

        for (int j = 0; j < degre; j++) {
            if (fscanf(frii, "%Le %d", &p, &e) != 2) {
                fprintf(stderr, "Erreur lecture Rii: arc (ligne %d)\n", i);
                exit(2);
            }
            if (pos < nnz) {
                col[pos] = e;
                val[pos] = p;
            }
            pos++;
        }
    }

    if (pos != nnz) {
        fprintf(stderr, "Attention: nnz lu = %d, nnz attendu (.sz) = %d\n", pos, nnz);
        if (pos > nnz) {
            fprintf(stderr, "Erreur: nnz lu > nnz attendu. Corrige .sz ou alloue dynamiquement.\n");
            exit(2);
        }
    }
}

/*
 * Écriture de la matrice réordonnée :
 * new_row = 0..n-1
 * old_row = new_to_old[new_row]
 * Colonnes remappées avec old_to_new[old_col], puis triées pour garder l'ordre croissant.
 */
static void write_reordered_Rii(const char *riiout_path,
                                int n,
                                const int *rowptr, const int *col, const long double *val,
                                const int *old_to_new, const int *new_to_old)
{
    FILE *fout = fopen(riiout_path, "w");
    if (!fout) die("fopen riiout");

    Edge *tmp = NULL;
    int tmp_cap = 0;

    for (int new_row = 0; new_row < n; new_row++) {
        int old_row = new_to_old[new_row];

        int start = rowptr[old_row];
        int end   = rowptr[old_row + 1];
        int degre = end - start;

        if (degre > tmp_cap) {
            tmp_cap = degre;
            tmp = (Edge*)realloc(tmp, (size_t)tmp_cap * sizeof(Edge));
            if (!tmp) { fprintf(stderr, "realloc tmp failed\n"); exit(2); }
        }

        for (int k = 0; k < degre; k++) {
            int old_col = col[start + k];
            tmp[k].col = old_to_new[old_col];
            tmp[k].val = val[start + k];
        }

        if (degre > 1) qsort(tmp, (size_t)degre, sizeof(Edge), cmp_edge_col);

        fprintf(fout, "%12d %12d ", new_row, degre);
        for (int k = 0; k < degre; k++) {
            fprintf(fout, "% .15LE%12d", tmp[k].val, tmp[k].col);
        }
        fprintf(fout, "\n");
    }

    free(tmp);
    fclose(fout);
}

/*
 * Lit model.pi (N lignes) et écrit model-reordre.pi en appliquant la permutation.
 *
 * Convention (alignée avec ton GTHCreux) :
 * - fichier pi : pi[1]..pi[N] (1-based), ligne k = probabilité de l'état old_id = k-1
 * - on produit un fichier pi réordonné : ligne (new_id+1) = probabilité de l'état new_id.
 */
static void reorder_pi_file(const char *pi_path, const char *piout_path,
                            int n, const int *old_to_new)
{
    FILE *fin = fopen(pi_path, "r");
    if (!fin) die("fopen pi input");

    FILE *fout = fopen(piout_path, "w");
    if (!fout) die("fopen pi output");

    double *pi_old = (double*)malloc((size_t)n * sizeof(double));
    double *pi_new = (double*)calloc((size_t)n, sizeof(double));
    if (!pi_old || !pi_new) die("malloc pi");

    // Lecture : N nombres (une ligne par nombre). On tolère espaces/retours.
    for (int i = 0; i < n; i++) {
        if (fscanf(fin, "%lf", &pi_old[i]) != 1) {
            fprintf(stderr, "Erreur: fichier .pi trop court (attendu %d valeurs), bloqué à i=%d\n", n, i);
            exit(2);
        }
    }

    // Permutation : old_id -> new_id
    for (int old_id = 0; old_id < n; old_id++) {
        int new_id = old_to_new[old_id];
        pi_new[new_id] = pi_old[old_id];
    }

    // Écriture : on garde le style GTHCreux (une proba par ligne, notation scientifique)
    for (int new_id = 0; new_id < n; new_id++) {
        fprintf(fout, " %.14e\n", pi_new[new_id]);
    }

    free(pi_old);
    free(pi_new);
    fclose(fin);
    fclose(fout);
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: ./reordonner <modelName> <city>  (ex: ./reordonner Rabat_a0 Rabat )\n");
        return 1;
    }

    char sz[256], cd[256], rii[256], pi[256];
    char szout[256], cdout[256], part[256], riiout[256], piout[256];

    snprintf(sz,     sizeof(sz),     "./%s/%s.sz",             argv[2],argv[1]);
    snprintf(cd,     sizeof(cd),     "./%s/%s.cd",             argv[2],argv[1]);
    snprintf(rii,    sizeof(rii),    "./%s/%s.Rii",            argv[2],argv[1]);
    snprintf(pi,     sizeof(pi),     "./%s/%s.pi",             argv[2],argv[1]);

    snprintf(szout,  sizeof(szout),  "./%s/%s-reordre.sz",     argv[2],argv[1]);
    snprintf(cdout,  sizeof(cdout),  "./%s/%s-reordre.cd",     argv[2],argv[1]);
    snprintf(part,   sizeof(part),   "./%s/%s-reordre.part",   argv[2],argv[1]);
    snprintf(riiout, sizeof(riiout), "./%s/%s-reordre.Rii",    argv[2],argv[1]);
    snprintf(piout,  sizeof(piout),  "./%s/%s-reordre.pi",     argv[2],argv[1]);

    FILE *fsz  = fopen(sz, "r");
    FILE *fcd  = fopen(cd, "r");
    FILE *frii = fopen(rii,"r");
    if (!fsz || !fcd || !frii) die("fopen input");

    FILE *fszout = fopen(szout,"w");
    FILE *fcdout = fopen(cdout,"w");
    FILE *fpart  = fopen(part,"w");
    if (!fszout || !fcdout || !fpart) die("fopen output");

    int nnz, n, dim;
    if (fscanf(fsz, "%d", &nnz) != 1) { fprintf(stderr, "Erreur lecture nnz\n"); return 2; }
    if (fscanf(fsz, "%d", &n)   != 1) { fprintf(stderr, "Erreur lecture n\n");   return 2; }
    if (fscanf(fsz, "%d", &dim) != 1) { fprintf(stderr, "Erreur lecture dim\n"); return 2; }

    if (n <= 0 || n > MAXETATS) {
        fprintf(stderr, "n invalide (%d). MAXETATS=%d\n", n, MAXETATS);
        return 2;
    }
    if (nnz < 0) {
        fprintf(stderr, "nnz invalide (%d)\n", nnz);
        return 2;
    }

    fprintf(fszout, "%12d \n%12d \n%12d \n", nnz, n, dim);

    // ---- Lire les états ----
    State *states = (State*)malloc((size_t)n * sizeof(State));
    if (!states) die("malloc states");

    for (int i = 0; i < n; i++) {
        int id, M, D, H, X;
        if (fscanf(fcd, "%d %d %d %d %d", &id, &M, &D, &H, &X) != 5) {
            fprintf(stderr, "Erreur lecture .cd ligne %d\n", i);
            return 2;
        }
        states[i].old_id = id;
        states[i].M = M; states[i].D = D; states[i].H = H; states[i].X = X;
    }

    // ---- Tri O(n log n) ----
    qsort(states, (size_t)n, sizeof(State), cmp_state);

    // ---- Écrire partitions (M,D) ----
    EcrirePartitions_MD_FILE_State(fpart, states, n);

    // ---- Construire permutations ----
    int *old_to_new = (int*)malloc((size_t)n * sizeof(int));
    int *new_to_old = (int*)malloc((size_t)n * sizeof(int));
    if (!old_to_new || !new_to_old) die("malloc perm");

    for (int newi = 0; newi < n; newi++) {
        int old = states[newi].old_id;
        if (old < 0 || old >= n) {
            fprintf(stderr, "Erreur: old_id=%d hors [0..%d]\n", old, n-1);
            return 2;
        }
        old_to_new[old] = newi;
        new_to_old[newi] = old;
    }

    // ---- Écrire .cd réordonné ----
    for (int i = 0; i < n; i++) {
        fprintf(fcdout, "%12d %12d %12d %12d %12d\n",
                i, states[i].M, states[i].D, states[i].H, states[i].X);
    }

    // ---- Lire Rii en CSR (1 passage) ----
    int *rowptr = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *col    = (int*)malloc((size_t)nnz * sizeof(int));
    long double *val = (long double*)malloc((size_t)nnz * sizeof(long double));
    if (!rowptr || !col || !val) die("malloc CSR");

    read_Rii_to_CSR(frii, n, nnz, rowptr, col, val);

    // ---- Écrire Rii réordonné ----
    write_reordered_Rii(riiout, n, rowptr, col, val, old_to_new, new_to_old);

    // ---- Réordonner pi : (optionnel mais demandé) ----
    // Si le fichier .pi n'existe pas, on avertit et on continue.
    FILE *testpi = fopen(pi, "r");
    if (!testpi) {
        fprintf(stderr, "Warning: fichier %s introuvable, je ne génère pas %s\n", pi, piout);
    } else {
        fclose(testpi);
        reorder_pi_file(pi, piout, n, old_to_new);
        printf("Pi re-ordered written to %s\n", piout);
    }

    // ---- Clean ----
    free(states);
    free(old_to_new);
    free(new_to_old);
    free(rowptr);
    free(col);
    free(val);

    fclose(fsz);
    fclose(fcd);
    fclose(frii);
    fclose(fszout);
    fclose(fcdout);
    fclose(fpart);

    printf("Matrix re-order done!\n");
    return 0;
}
