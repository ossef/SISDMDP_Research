import numpy as np
from gth_full import gth_full

def algo_chiu(P_sparse, superstates):
    """
    Version sparse-compatible de l'algorithme de Feinberg & Chiu (1987).
    Hypothèse : le single-input est le PREMIER dans chaque super-état.
    Inter-superstates : GTH
    Intra-superstates : GTH
    """

    #start = time.time()
    n = P_sparse.shape[0]
    K = len(superstates)

    phi_list = []
    psi_list = []

    # Étape 1 : vecteurs stationnaires locaux
    for r in range(K):
        S_r = [int(s) for s in superstates[r]]
        nr = len(S_r)

        # Sous-matrice locale Pr : extraction sparse puis conversion temporaire en dense
        Pr = P_sparse[S_r, :][:, S_r].tocoo().toarray()

        # Ajout des transitions externes vers le single-input (indice 0)
        for i, i_global in enumerate(S_r):
            not_in_Sr = list(set(range(n)) - set(S_r))
            Pr[i, 0] += P_sparse[i_global, not_in_Sr].sum()

        # Solving inter-superstates with gth_full
        Pi_r = gth_full(Pr)
        phi_list.append(Pi_r[1:])  # internes
        psi_list.append(Pi_r[0])   # single-input

    # Étape 2 : construction de la matrice entre super-états P^
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

            prob = 0.0
            for idx, state in enumerate(internal_i):
                prob += phi[idx] * P_sparse[state, sj]
            prob += psi * P_sparse[si, sj]
            Pk[i, j] = prob

        Pk[i, i] = 1.0 - np.sum(Pk[i, :])

    # Étape 3 : Solving intra-superstates with GTH
    alpha = gth_full(Pk)

    # Étape 4 : reconstruction de π
    pi = np.zeros(n)
    for r in range(K):
        S_r = superstates[r]
        si = S_r[0]
        internal = S_r[1:]
        phi = phi_list[r]
        psi = psi_list[r]
        pi[si] = alpha[r] * psi
        for idx, g_idx in enumerate(internal):
            pi[g_idx] = alpha[r] * phi[idx]

    #duration = time.time() - start
    #print("\nπ global (stationnaire) :", pi)
    #print("Somme des composantes   :", np.sum(pi))
    #print(f"Durée de algo chiu classique : {duration:.4f} sec")
    #np.savetxt("result.chiu.sparse.pi", pi, fmt="%.18e")
    return pi


"""
#Test Chiu Classic 

P = [
    [0.39, 0.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.08],  # état 1
    [0.00, 0.00, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.13],  # état 2
    [0.00, 0.07, 0.31, 0.00, 0.00, 0.00, 0.00, 0.42, 0.20, 0.00],  # état 3
    [0.00, 0.00, 0.00, 0.23, 0.62, 0.00, 0.00, 0.00, 0.00, 0.15],  # état 4
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],  # état 5
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.17, 0.54, 0.29, 0.00, 0.00],  # état 6
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],  # état 7
    [0.41, 0.20, 0.39, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # état 8
    [0.00, 0.00, 0.00, 0.25, 0.65, 0.00, 0.00, 0.00, 0.00, 0.10],  # état 9
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00],  # état 10
]
superstates=[[8.00, 1.00 , 2.00 , 3.00 ], [9.00, 4.00 , 5.00 ], [10.00, 6.00 , 7.00 ]]
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
