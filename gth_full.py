import numpy as np
import time

def gth_full(P):
    """
    Implémentation robuste de l'algorithme GTH pour matrice de transition P (modifiée en place).
    P doit être une copie si l'on souhaite conserver la matrice initiale.
    """
    #start= time.time()
    #P = P.copy()
    nr = P.shape[0] - 1

    for n in range(nr, -1, -1):  # de nr à 0
        S = np.sum(P[n, :n])
        for i in range(n):
            P[i, n] = P[i, n] / S
        for i in range(n):
            for j in range(n):
                P[i, j] += P[i, n] * P[n, j]

    pi = np.zeros(nr + 1)
    pi[0] = 1.0
    Tot = 1.0

    for j in range(1, nr + 1):
        pi[j] = P[0, j]
        for k in range(1, j):
            pi[j] += pi[k] * P[k, j]
        Tot += pi[j]

    pi /= Tot
    #duree= time.time() - start
    #print(f"duree gth full: {duree: .4f} seconds", duree)
    return pi
