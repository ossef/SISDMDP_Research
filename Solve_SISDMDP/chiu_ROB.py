import numpy as np
from gth_full import gth_full
#import time, sys

def steady_State_ROB_B(matrice_transition, N):
    """Distribution stationnaire pour structure de type B (root state à l'indice 0)."""
    alpha = np.zeros(N)
    Pi = np.zeros(N)
    alpha[0] = 1

    for q in range(1, N):
        colonne_q = matrice_transition.getcol(q)
        alpha[q] = (alpha[:q].dot(colonne_q[:q].toarray().ravel())) / (1 - matrice_transition[q, q])

    Pi[0] = 1 / (1 + np.sum(alpha[1:]))
    Pi[1:] = alpha[1:] * Pi[0]

    if not np.isclose(np.sum(Pi), 1.0, atol=1e-10):
        raise ValueError(f"Somme ≠ 1 dans Pi : {np.sum(Pi)}")

    return Pi

def chiu_ROB_B(P, superstates):
    """
    Algorithme de Feinberg & Chiu (1987) compatible avec matrice sparse CSR.
    Hypothèse : le single-input state est en premier.
    Inter-superstates : Rob-B
    Intra-superstates : GTH
    """
    #start = time.time()
    K = len(superstates)
    n = P.shape[0]

    phi_list, psi_list = [], []

    # Étape 1 : Vecteurs stationnaires locaux
    for r, S_r in enumerate(superstates):
        #indices = [int(s) for s in S_r]
        indices = S_r
        nr = len(indices)

        # Ajout des transitions externes vers le single-input
        not_in_Sr = np.setdiff1d(np.arange(n), indices, assume_unique=True)
        Pr = P[indices][:, indices].tolil()
        # Boucle sur les ajouts
        for i, i_global in enumerate(indices):
            external_sum = P[i_global, not_in_Sr].sum()
            Pr[i, 0] += external_sum

        # Solving inter-superstates with Rob-B
        Pr = Pr.tocsr()  
        Pi_r = steady_State_ROB_B(Pr, nr)

        phi_list.append(Pi_r[1:])
        psi_list.append(Pi_r[0])

    # Étape 2 : construction de la matrice P^ entre super-états
    Pk = np.zeros((K, K))
    for i in range(K):
        S_i = superstates[i]
        si = S_i[0]
        internal_i = S_i[1:]
        phi = phi_list[i]
        psi = psi_list[i]

        for j in range(K):
            if i == j:
                continue

            sj = superstates[j][0]

            # Produit phi · P[state, sj] (en bloc)
            P_col_sj = P[:, sj]
            P_internal = P_col_sj[internal_i].toarray().flatten()
            prob = np.dot(phi, P_internal) + psi * P[si, sj]

            Pk[i, j] = prob

        Pk[i, i] = 1.0 - np.sum(Pk[i, :])

    # Étape 3 : Solving intra-superstates with GTH
    alpha = gth_full(Pk)

    # Étape 4 : reconstruction du vecteur π
    pi = np.zeros(n)
    for r in range(K):
        S_r = superstates[r]
        si = S_r[0]
        internal = S_r[1:]
        phi = phi_list[r]
        psi = psi_list[r]

        all_indices = np.array([si] + internal)
        all_weights = np.concatenate(([psi], phi)) * alpha[r]

        pi[all_indices] = all_weights

    #duration = time.time() - start
    #print("\nπ global (stationnaire) :", pi)
    #print("Somme des composantes :", np.sum(pi))
    #print(f"Durée de CHIU_ROB_B (sparse-compatible) : {duration:.4f} sec")
    #np.savetxt("result.chiu.RB.pi", pi, fmt="%.18e")
    return pi


"""
#Test Chiu Rob-B

S21 = [[0.39, 0.26, 0, 0 ,0 ,0 ,0, 0.27, 0, 0.08], #loop in this state, a "non_R_state"
      [0, 0, 0.82, 0, 0, 0, 0, 0, 0.05, 0.13], 
      [0, 0, 0.31, 0, 0, 0, 0,  0.49, 0.20, 0 ],  #loop in this state, an "R_state"
      [0, 0, 0, 0.23, 0.62, 0, 0, 0, 0, 0.15],    #loop in this state, a "non_R_state"
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0.17, 0.54, 0.29, 0, 0],    #loop in this state
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0.41, 0.20, 0.39, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0.25, 0.65, 0, 0, 0, 0, 0.10],
      [0, 0, 0, 0, 0, 0.5, 0.4, 0, 0, 0.1]        #loop in this state, a "non_R_state" (also superState)
      ]
Parts_S21 = [[7, 0, 1, 2], [9, 5, 6], [8, 3, 4]]
"""


