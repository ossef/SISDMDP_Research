import numpy as np
import time
from scipy.sparse import vstack, isspmatrix_csr
from chiu_ROB import chiu_ROB_B as steady_State_Chiu_Rob
from chiu_classic import algo_chiu as steady_State_Chiu
from collections import deque


MAX_ITER = 1e5                # Max Iterations for average and discounted
EPSILON  = 1e-15              # Precision for convergence of iterative algorithms
STAGNATION_WINDOW = 100       # Only for average : RVI, RPI+FP
STAGNATION_THRESHOLD = 1e-13  # Only for average : RVI, RPI+FP

#------------------------------------- Utils----------------------------------------------------#

def Gauss_Jordan_solver(G) :
    # II - Full Gauss Jordan solver of GX = 0
    # The same resolution as average reward criteria
    N = len(G)
    #print("GJ for N = ",N)
    x = np.zeros(N)
    for col in range(N):
        # Trouver le meilleur pivot
        pivot = -0.1
        pivot_row = -1
        for row in range(col, N):
            if abs(G[row, col]) > pivot:
                pivot = abs(G[row, col])
                pivot_row = row

        # Vérifier si la solution peut être trouvée
        if pivot <= 1e-5:
            raise("Erreur dans l'évaluation de la politique GJ. Matrice singuliere.")

        # Échanger les lignes pour utiliser le meilleur pivot
        if pivot_row != col:
            G[[col, pivot_row]] = G[[pivot_row, col]]

        # Effectuer l'élimination
        for row1 in range(N):
            if row1 != col:
                factor = G[row1, col] / G[col, col]
                G[row1, col:] -= factor * G[col, col:]

    # Trouver la solution
    for row in range(N):
        x[row] = G[row, N] / G[row, row]

    return x #vecteur de valeurs

def get_partition_state_roles(partition, superstates, P):
    R_states = []
    for i in partition:
        row_start, row_end = P.indptr[i], P.indptr[i+1]
        successors = P.indices[row_start:row_end]
        successors = [s for s in successors if s != i]
        if all(s in superstates for s in successors):
            R_states.append(i)
    non_R_states = [i for i in partition if i not in R_states]
    superstate = partition[0]
    return superstate, R_states, list(reversed(non_R_states))

def policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor ): 
    new_policy = policy.copy()

    Q = np.zeros((N, A))
    for a in range(A):
        Q[:, a] = np.round(Ria[:, a] + d_factor*P[a].dot(H),20)  # à mettre à 10
    new_policy= np.argmax(Q, axis=1)

    """
    # Amélioration de la politique avec critère de sélection
    for s in range(N):
        max_q_value = np.max(Q[s, :])
        # Sélectionner les actions dont la valeur Q est proche du max, avec une tolérance epsilon
        close_actions = np.where(np.abs(Q[s, :] - max_q_value) <= 1e-10)[0]
        # Choisir l'action avec l'indice le plus faible parmi celles proches du max
        new_policy[s] = np.min(close_actions)
    """

    return new_policy


'''------------------------------------- I) AVG criteria--------------------------------------------'''

#------------------ Policy Evaluation

def steady_State_Power(matrice_transition):
    N = matrice_transition.shape[0]
    pi = np.ones(N) / N
    it = 0

    for it in range(int(1e5)):
        pi_old = np.copy(pi)
        pi = pi @ matrice_transition  # produit matriciel sparse/dense
        norme = np.linalg.norm(pi - pi_old, ord=2)

        if norme < 1e-15:
            print("Iterations =", it+1, "et norme =", norme)
            return pi

    raise ValueError("Méthode de puissance n'a pas convergé à epsilon.")

def average_Reward_Power(matrice_transition, R):
    pi = steady_State_Power(matrice_transition)
    rau = np.sum(pi * R)
    return rau

def average_Reward_Chiu(matrice_transition, Parts, R):
    pi = steady_State_Chiu(matrice_transition, Parts)
    rau = np.sum(pi * R)
    return rau

def average_Reward_Chiu_Rob(matrice_transition, Parts, R):
    pi = steady_State_Chiu_Rob(matrice_transition, Parts)
    rau = np.sum(pi * R)
    return rau

def average_policy_evaluation_FP(P, Ria, policy, N, relative_state=0):
    #print("@@@@ AVG Relative Policy Evaluation with FP (sparse) @@@@")
    H = np.zeros(N)
    k = 0
    #start_time = time.time()
    span_history = deque(maxlen=STAGNATION_WINDOW)

    for k in range(int(MAX_ITER)):
        H_prev = H.copy()

        for s in range(N):
            a = policy[s]
            P_a = P[a]  # Get sparse matrix for action a
            row_start, row_end = P_a.indptr[s], P_a.indptr[s+1]
            indices = P_a.indices[row_start:row_end]
            data = P_a.data[row_start:row_end]
            H[s] = Ria[s, a] + np.dot(data, H_prev[indices])

        rau = H[relative_state]
        H -= rau

        diff = H - H_prev
        span = np.max(diff) - np.min(diff)

        print("Avg-FP : Iterations = ", k+1, "| span = ", span, " |rau = ",rau)
        if span < EPSILON:
            break

        # Stocker l'historique du span
        span_history.append(span)

        # Critère de stagnation
        if len(span_history) == STAGNATION_WINDOW:
            if max(span_history) - min(span_history) < STAGNATION_THRESHOLD:
                print("Convergence: span stagnation")
                break

    #ProcessTime = time.time() - start_time
    print("Avg-FP : Iterations = ", k+1, "| span = ", span, " |rau = ",rau)
    #print("---> @@@ Processing time for FP algorithm = {:.4f} s".format(ProcessTime))
    return rau, H

def average_policy_evaluation_GJ(P, Ria, policy, N, relative_state=0):
    #print("@@@@ AVG Policy evaluation with Gauss-Jordan (adaptive) @@@@@")
    #start_time = time.time()
    
    # Cas 1: multi-actions (P est une liste/array de matrices par action)
    if isinstance(P, (list, np.ndarray)) and hasattr(P[0], "getrow"):
        G = np.zeros((N, N + 1))
        for row in range(N):
            a = policy[row]
            P_a = P[a]

            if isspmatrix_csr(P_a):
                row_start = P_a.indptr[row]
                row_end = P_a.indptr[row + 1]
                cols = P_a.indices[row_start:row_end]
                data = P_a.data[row_start:row_end]

                found_diag = False
                for idx, col in enumerate(cols):
                    if row == col:
                        G[row, col] = 1.0 - data[idx]
                        found_diag = True
                    else:
                        G[row, col] = -data[idx]
                # Forcer la diagonale si absente
                if not found_diag:
                    G[row, row] = 1.0
                # Colonne spéciale pour rho
                G[row, relative_state] = 1.0

            else:
                # Rare cas: matrice dense dans P[a]
                for col in range(N):
                    if row == col:
                        G[row, col] = 1 - P_a[row, col]
                    else:
                        G[row, col] = - P_a[row, col]

        G[:, -1] = Ria[np.arange(N), policy]

    # Cas 2: P est une unique matrice (dense ou sparse)
    else:
        N = len(P)
        G = np.zeros((N, N + 1))
        for row in range(N):
            for col in range(N):
                if col == row:
                    G[row, col] = 1.0 - P[row, col]
                else:
                    G[row, col] = -P[row, col]
            # Colonne spéciale pour rho
            G[row, relative_state] = 1.0

        # Dernière colonne = Ria
        G[:, -1] = Ria

    # Résolution via Gauss-Jordan
    x = Gauss_Jordan_solver(G)

    rau = x[relative_state]
    x[relative_state] = 0.0  # Normalisation
    #ProcessTime = time.time() - start_time
    #print("---> @@@ Processing time for GJ algorithm = {:.4f} s".format(ProcessTime))

    return rau, x

def average_policy_evaluation_SISDMCSC_CHIU(P, Ria, policy, N, Parts, relative_state):
    #print("\n@@@@ AVG Policy evaluation for SISCSDMC (sparse)  @@@@@")
    #start_time = time.time()

    nParts = len(Parts)
    superstates = [part[0] for part in Parts]
    superstate_to_idx = {s: i for i, s in enumerate(superstates)}
    M_partitions = []
    b_partitions = []
    state_roles_partitions = []

    # --------> 0) cas AVG reward ---------
    P_policy = vstack([P[policy[s]].getrow(s) for s in range(N)]).tocsr()
    R_policy = Ria[np.arange(N), policy].astype(float)
    average = average_Reward_Chiu(P_policy, Parts, R_policy)
    #average = average_Reward_Power(P_policy, R_policy)
    R_policy -= average
    diag = np.ones(N) - P_policy.diagonal()

    # --------> A) Substitution locale dans chaque partition ------------
    for p in range(nParts):
        partition = Parts[p]
        part_size = len(partition)
        global_to_local = {partition[i]: i for i in range(part_size)}
        M = np.zeros((part_size, nParts))
        b = np.zeros(part_size)

        superstate, R_states, non_R_states_ordered = get_partition_state_roles(partition, superstates, P_policy)
        state_roles_partitions.append((superstate, R_states, non_R_states_ordered))

        R_states_local = [global_to_local[state] for state in R_states]

        # Étape 1 : R_states → superstates
        for i in R_states_local:
            state_i = partition[i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != state_i and k in superstate_to_idx:
                    s_idx = superstate_to_idx[k]
                    M[i, s_idx] = row.data[idx]
            M[i] /= diag[state_i]  #Loops treatment
            b[i] /= diag[state_i]  #Loops treatment

        # Étape 2 : non_R_states
        for state_i in non_R_states_ordered:
            i = global_to_local[state_i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                prob = row.data[idx]
                if k != state_i:
                    if k not in superstate_to_idx:
                        k_local = global_to_local[k]
                        M[i] += prob * M[k_local]
                        b[i] += prob * b[k_local]
                    else:
                        col_k = superstate_to_idx[k]
                        M[i, col_k] += prob
            M[i] /= diag[state_i]
            b[i] /= diag[state_i]

        M_partitions.append(M)
        b_partitions.append(b)

    # -----------> B) Assemblage global -----------------------
    A = np.vstack([M[0] for M in M_partitions])
    B = np.array([b[0] for b in b_partitions])

    # ----------> C) Résolution du système  --------------------
    #The relative state we choose in the sub-sytem is '0' that corresponds to superState '7'
    #Be carreful to this when comparing algorithms. Otherwise, relative will not be the same
    _, S = average_policy_evaluation_GJ(A, B, None, N, 0)
    #print("S = ", S)

    # ----------> D) Injection finale (optimized version using M and b) --------------------------
    V_global = np.zeros(N)

    # Inject known superstate values from the global solution
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]  # S[p] = value from system resolution

    # Inject all other values using local systems
    for p in range(nParts):
        partition = Parts[p]
        M = M_partitions[p]
        b = b_partitions[p]
        V_sup = S  # or V_global[superstates]

        # Reconstruct local values
        V_local = M @ V_sup + b

        # Inject only non-superstate values
        for i, state in enumerate(partition):
            if state != superstates[p]:  # avoid overwriting already-set superstate
                V_global[state] = V_local[i]

    """
    # ----------> D) Injection finale --------------------------
    V_global = np.zeros(N)

    # 1. superStates
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]

    for p in range(nParts):
        superstate, R_states, non_R_states_ordered = state_roles_partitions[p]

        # 2. R_states
        for i in R_states:
            row = P_policy.getrow(i)
            V_global[i] = R_policy[i]
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != i and k in superstate_to_idx:
                    V_global[i] += row.data[idx] * V_global[k]
            V_global[i] /= diag[i]

        # 3. non_R_states
        for i in non_R_states_ordered:
            if i != superstate:
                row = P_policy.getrow(i)
                V_global[i] = R_policy[i]
                for idx in range(row.indptr[0], row.indptr[1]):
                    k = row.indices[idx]
                    if k != i:
                        V_global[i] += row.data[idx] * V_global[k]
                V_global[i] /= diag[i]
    """

    #print("@@@ Processing time for SISCSDMC algorithm = {:.4f} seconds".format(time.time() - start_time))
    return average, V_global

def average_policy_evaluation_SISDMCSC_CHIU_ROB(P, Ria, policy, N, Parts, relative_state):
    #print("\n@@@@ AVG Policy evaluation for SISCSDMC (sparse)  @@@@@")
    #start_time = time.time()

    nParts = len(Parts)
    superstates = [part[0] for part in Parts]
    superstate_to_idx = {s: i for i, s in enumerate(superstates)}
    M_partitions = []
    b_partitions = []
    state_roles_partitions = []

    # --------> 0) cas AVG reward ---------
    P_policy = vstack([P[policy[s]].getrow(s) for s in range(N)]).tocsr()
    R_policy = Ria[np.arange(N), policy].astype(float)
    average = average_Reward_Chiu_Rob(P_policy, Parts, R_policy)
    #average = average_Reward_Power(P_policy, R_policy)
    R_policy -= average
    diag = np.ones(N) - P_policy.diagonal()

    # --------> A) Substitution locale dans chaque partition ------------
    for p in range(nParts):
        partition = Parts[p]
        part_size = len(partition)
        global_to_local = {partition[i]: i for i in range(part_size)}
        M = np.zeros((part_size, nParts))
        b = np.zeros(part_size)

        superstate, R_states, non_R_states_ordered = get_partition_state_roles(partition, superstates, P_policy)
        state_roles_partitions.append((superstate, R_states, non_R_states_ordered))

        R_states_local = [global_to_local[state] for state in R_states]

        # Étape 1 : R_states → superstates
        for i in R_states_local:
            state_i = partition[i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != state_i and k in superstate_to_idx:
                    s_idx = superstate_to_idx[k]
                    M[i, s_idx] = row.data[idx]
            M[i] /= diag[state_i]  #Loops treatment
            b[i] /= diag[state_i]  #Loops treatment

        # Étape 2 : non_R_states
        for state_i in non_R_states_ordered:
            i = global_to_local[state_i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                prob = row.data[idx]
                if k != state_i:
                    if k not in superstate_to_idx:
                        k_local = global_to_local[k]
                        M[i] += prob * M[k_local]
                        b[i] += prob * b[k_local]
                    else:
                        col_k = superstate_to_idx[k]
                        M[i, col_k] += prob
            M[i] /= diag[state_i]
            b[i] /= diag[state_i]

        M_partitions.append(M)
        b_partitions.append(b)

    # -----------> B) Assemblage global -----------------------
    A = np.vstack([M[0] for M in M_partitions])
    B = np.array([b[0] for b in b_partitions])

    # ----------> C) Résolution du système  --------------------
    #The relative state we choose in the sub-sytem is '0' that corresponds to superState '7'
    #Be carreful to this when comparing algorithms. Otherwise, relative will not be the same
    _, S = average_policy_evaluation_GJ(A, B, None, N, 0)
    #print("S = ", S)

    # ----------> D) Injection finale (optimized version using M and b) --------------------------
    V_global = np.zeros(N)

    # Inject known superstate values from the global solution
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]  # S[p] = value from system resolution

    # Inject all other values using local systems
    for p in range(nParts):
        partition = Parts[p]
        M = M_partitions[p]
        b = b_partitions[p]
        V_sup = S  # or V_global[superstates]

        # Reconstruct local values
        V_local = M @ V_sup + b

        # Inject only non-superstate values
        for i, state in enumerate(partition):
            if state != superstates[p]:  # avoid overwriting already-set superstate
                V_global[state] = V_local[i]

    """
    # ----------> D) Injection finale --------------------------
    V_global = np.zeros(N)

    # 1. superStates
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]

    for p in range(nParts):
        superstate, R_states, non_R_states_ordered = state_roles_partitions[p]

        # 2. R_states
        for i in R_states:
            row = P_policy.getrow(i)
            V_global[i] = R_policy[i]
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != i and k in superstate_to_idx:
                    V_global[i] += row.data[idx] * V_global[k]
            V_global[i] /= diag[i]

        # 3. non_R_states
        for i in non_R_states_ordered:
            if i != superstate:
                row = P_policy.getrow(i)
                V_global[i] = R_policy[i]
                for idx in range(row.indptr[0], row.indptr[1]):
                    k = row.indices[idx]
                    if k != i:
                        V_global[i] += row.data[idx] * V_global[k]
                V_global[i] /= diag[i]
    """

    #print("@@@ Processing time for SISCSDMC algorithm = {:.4f} seconds".format(time.time() - start_time))
    return average, V_global


#---------------- A- Relative Policy Iteration Algorithm (three versions)

def average_policy_Iteration_Csr_FP(P, Ria, N, A, relative_state): # (RPI + FP) Relative Policy Iteration, using Fixed point approx 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ AVG: RPI + FP algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        rau, H = average_policy_evaluation_FP(P, Ria, policy, N, relative_state)
        #print("rau = {}, Policy {} ".format(rau,policy))
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor =1)
        #print("new_Policy = ",new_policy)
        
        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        #print(f'---> Iteration k = {k}, rau = {rau}')
        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for AVG-RPI-FP algorithm = {} (s) ".format(ProcessTime))
    return rau, policy, ProcessTime, k

def average_policy_Iteration_Csr_GJ(P, Ria, N, A, relative_state): # (RPI + GJ) Relative Policy Iteration, using Gauss-Jordan elimination 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ AVG: RPI + GJ algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        rau, H = average_policy_evaluation_GJ(P, Ria, policy, N, relative_state)
        #print("rau = {}, Policy {} ".format(rau,policy))
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor =1)
        #print("new_Policy = ",new_policy)
        
        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for AVG-RPI-GJ algorithm = {} (s) ".format(ProcessTime))
    return rau, policy, ProcessTime, k

def average_policy_Iteration_Csr_Chiu(P, Ria, N, A, Parts, relative_state): # (MRPI + Chiu) Proposed Policy Iteration, SISDMDP structure 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ AVG: MRPI + Chiu algorithm : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        rau, H = average_policy_evaluation_SISDMCSC_CHIU(P, Ria, policy, N, Parts, relative_state)
        #print("rau = {}, Policy {} ".format(rau,policy))
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor = 1)
        #print("new_Policy = ",new_policy)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for AVG-MRPI-Chiu algorithm = {} (s) ".format(ProcessTime))
    return rau, policy, ProcessTime, k

def average_policy_Iteration_Csr_Chiu_Rob(P, Ria, N, A, Parts, relative_state): # (MRPI + Chiu+ RobB) Proposed Policy Iteration, SISDMDP structure 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ AVG: MRPI + Chiu + Rob algorithm : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        rau, H = average_policy_evaluation_SISDMCSC_CHIU_ROB(P, Ria, policy, N, Parts, relative_state)
        #print("rau = {}, Policy {} ".format(rau,policy))
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor = 1)
        #print("new_Policy = ",new_policy)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for AVG-MRPI_Ciu_Rob algorithm = {} (s) ".format(ProcessTime))
    return rau, policy, ProcessTime, k


#---------------- B- Relative Value Iteration Algorithm 

def average_relative_Value_Iteration_Csr(P, Ria, N, A, relative_state) :  # (RVI) Relative Value Iteration for "Sparse CSR" actions matrixes   
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ AVG: RVI algorithm  :  Sparse Matrix version @@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
    k = 0 

    span_history = deque(maxlen=STAGNATION_WINDOW)
    rau_history  = deque(maxlen=STAGNATION_WINDOW)

    while(k <= MAX_ITER):
        k += 1
        #J_prev = np.copy(J)
        J_prev = J.copy()

        Q = np.zeros((N, A))
        for a in range(A):
            Q[:, a] = Ria[:, a] + P[a].dot(J_prev)   #Q[:, a] = Ria[:, a] + P[a].dot(J_prev)
        J = np.max(Q, axis=1)

        rau = J[relative_state]
        J -= rau
        J[relative_state] = 0

        diff = J - J_prev
        span = max(diff) - min(diff)

        print("Iteration k = {}, rau = {}, span = {:.15e}".format(k,rau,span))
        #print("rau RVI =", rau)
        # Vérifier la précision
        if span < EPSILON :
            break
        
        # Stocker l'historique du span
        span_history.append(span)
        rau_history.append(round(rau,15))

        # Critère de stagnation
        if len(span_history) == STAGNATION_WINDOW:
            if max(span_history) - min(span_history) < STAGNATION_THRESHOLD or max(rau_history)-min(rau_history)<STAGNATION_THRESHOLD:
                print("Convergence: Span stagnation ")
                break

    Optimal_Policy = np.argmax(Q, axis=1)
    print("Iteration k = {}, rau = {}, span = {:.15e}".format(k,rau,span))
    #print("J = ",J)
    #print("policy =",Optimal_Policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for AVG-RVI algorithm = {} (s) ".format(ProcessTime))
    return rau, Optimal_Policy, ProcessTime, k

'''------------------------------------- II) DSC criteria--------------------------------------------'''

#------------------ Policy Evaluation

def discount_policy_evaluation_FP(P, Ria, d_factor, policy, N):
    #print("\n@@@@ DSC Policy Evaluation with FP (sparse) @@@@")
    H = np.zeros(N)
    k = 0
    #start_time = time.time()

    for k in range(int(MAX_ITER)):
        H_prev = H.copy()

        for s in range(N):
            a = policy[s]
            P_a = P[a]  # Get sparse matrix for action a
            row_start, row_end = P_a.indptr[s], P_a.indptr[s+1]
            indices = P_a.indices[row_start:row_end]
            data = P_a.data[row_start:row_end]
            H[s] = Ria[s, a] + d_factor * np.dot(data, H_prev[indices])

        norme = np.max(np.abs(H - H_prev))
        if norme < EPSILON:
            break

    #ProcessTime = time.time() - start_time
    print("Iterations = ", k+1, " | norme = ", norme)
    #print("@@@ Processing time for FP algorithm = {:.4f} s".format(ProcessTime))
    return H

def discount_policy_evaluation_GJ(P, Ria, d_factor, policy, N):
    """
    P: soit une unique matrice (dense ou sparse), soit une liste de matrices (sparse) de shape (A,)
    Ria: (N, A)
    policy: vecteur (N,) donnant l'action choisie pour chaque état
    """
    #print("\n@@@@ DSC Policy evaluation with Gauss-Jordan (adaptive) @@@@@")

    #start_time = time.time()

    # Cas 1: multi-actions (P est une liste/array de matrices par action)
    if isinstance(P, (list, np.ndarray)) and hasattr(P[0], "getrow"):
        G = np.zeros((N, N + 1))
        for row in range(N):
            a = policy[row]
            P_a = P[a]

            if isspmatrix_csr(P_a):
                row_start = P_a.indptr[row]
                row_end = P_a.indptr[row + 1]
                cols = P_a.indices[row_start:row_end]
                data = P_a.data[row_start:row_end]

                found_diag = False
                for idx, col in enumerate(cols):
                    if row == col:
                        G[row, col] = 1 - d_factor * data[idx]
                        found_diag = True
                    else:
                        G[row, col] = -d_factor * data[idx]
                if not found_diag:
                    G[row, row] = 1.0
            else:
                # Rare cas: matrice dense dans P[a]
                for col in range(N):
                    if row == col:
                        G[row, col] = 1 - d_factor * P_a[row, col]
                    else:
                        G[row, col] = -d_factor * P_a[row, col]

        G[:, -1] = Ria[np.arange(N), policy]

    # Cas 2: P est une unique matrice (dense ou sparse)
    else:
        N = len(P)
        G = np.zeros((N, N + 1))
        for row in range(N):
            for col in range(N):
                if row == col:
                    G[row, col] = 1 - d_factor * P[row, col]
                else:
                    G[row, col] = -d_factor * P[row, col]

        # Dans ce cas, Ria est supposé être un vecteur (déjà réduit via la politique)
        G[:, -1] = Ria

    # Résolution du système linéaire
    x = Gauss_Jordan_solver(G)

    #elapsed = time.time() - start_time
    #print("@@@ Processing time for GJ algorithm = {:.4f} s".format(elapsed))
    return x

def discount_policy_evaluation_SISDMCSC(P, Ria, d_factor, policy, N, Parts):
    """
    P: list of sparse matrices (len A), each of shape (N, N)
    Ria: ndarray of shape (N, A)
    Parts: list of list of state indices (partitions)
    policy: array of shape (N,) — deterministic policy
    N: number of states
    """
    #print("\n@@@@ DSC Policy evaluation for SISCSDMC (sparse, optimized)  @@@@@")
    #start_time = time.time()

    nParts = len(Parts)
    superstates = [part[0] for part in Parts]
    superstate_to_idx = {s: i for i, s in enumerate(superstates)}
    M_partitions = []
    b_partitions = []
    state_roles_partitions = []

    # --------> 0) Reconstruct P_policy (sparse) efficiently from P and policy ---------
    P_policy = vstack([P[policy[s]].getrow(s) for s in range(N)]).tocsr()
    P_policy.data *= d_factor     #P_policy *= d_factor
    R_policy = Ria[np.arange(N), policy].astype(float)
    diag = np.ones(N) - P_policy.diagonal()

    # --------> A) Local substitution for each partition ----------
    for p in range(nParts):
        partition = Parts[p]
        part_size = len(partition)
        global_to_local = {partition[i]: i for i in range(part_size)}
        M = np.zeros((part_size, nParts))
        b = np.zeros(part_size)

        superstate, R_states, non_R_states_ordered = get_partition_state_roles(
            partition, superstates, P_policy
        )
        state_roles_partitions.append((superstate, R_states, non_R_states_ordered))

        R_states_local = [global_to_local[s] for s in R_states]

        # Étape 1 : R_states → superstates
        for i in R_states_local:
            state_i = partition[i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != state_i and k in superstate_to_idx:
                    s_idx = superstate_to_idx[k]
                    M[i, s_idx] = row.data[idx]
            M[i] /= diag[state_i]  #Loops treatment
            b[i] /= diag[state_i]  #Loops treatment

        # Étape 2 : non_R_states
        for state_i in non_R_states_ordered:
            i = global_to_local[state_i]
            b[i] = R_policy[state_i]
            row = P_policy.getrow(state_i)
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                prob = row.data[idx]
                if k != state_i:
                    if k not in superstate_to_idx:
                        k_local = global_to_local[k]
                        M[i] += prob * M[k_local]
                        b[i] += prob * b[k_local]
                    else:
                        col_k = superstate_to_idx[k]
                        M[i, col_k] += prob
            M[i] /= diag[state_i]
            b[i] /= diag[state_i]

        M_partitions.append(M)
        b_partitions.append(b)

    # -----------> B) Global assembly -----------------------
    A = np.vstack([M[0] for M in M_partitions])
    B = np.array([b[0] for b in b_partitions])

    # ----------> C) Solve linear system --------------------
    S = discount_policy_evaluation_GJ(A, B, 1, None, N)

    # ----------> D) Injection finale (optimized version using M and b) --------------------------
    V_global = np.zeros(N)

    # Inject known superstate values from the global solution
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]  # S[p] = value from system resolution

    # Inject all other values using local systems
    for p in range(nParts):
        partition = Parts[p]
        M = M_partitions[p]
        b = b_partitions[p]
        V_sup = S  # or V_global[superstates]

        # Reconstruct local values
        V_local = M @ V_sup + b

        # Inject only non-superstate values
        for i, state in enumerate(partition):
            if state != superstates[p]:  # avoid overwriting already-set superstate
                V_global[state] = V_local[i]

    """
    # ----------> D) Final injection ------------------------
    V_global = np.zeros(N)

    # 1. superStates
    for p, superstate in enumerate(superstates):
        V_global[superstate] = S[p]

    for p in range(nParts):
        superstate, R_states, non_R_states_ordered = state_roles_partitions[p]

        # 2. R_states
        for i in R_states:
            row = P_policy.getrow(i)
            V_global[i] = R_policy[i]
            for idx in range(row.indptr[0], row.indptr[1]):
                k = row.indices[idx]
                if k != i and k in superstate_to_idx:
                    V_global[i] += row.data[idx] * V_global[k]
            V_global[i] /= diag[i]

        # 3. non_R_states
        for i in non_R_states_ordered:
            if i != superstate:
                row = P_policy.getrow(i)
                V_global[i] = R_policy[i]
                for idx in range(row.indptr[0], row.indptr[1]):
                    k = row.indices[idx]
                    if k != i:
                        V_global[i] += row.data[idx] * V_global[k]
                V_global[i] /= diag[i]
    """

    #print("@@@ Processing time for SISCSDMC algorithm = {:.4f} seconds".format(time.time() - start_time))
    return V_global


#------------------ A - Policy Iteration Algorithm (three versions)

def discount_policy_Iteration_Csr_FP(P, Ria, N, A, d_factor): # (PI + FP) Policy Iteration, using Fixed point approx 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ DSC: PI + FP algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        H = discount_policy_evaluation_FP(P, Ria, d_factor, policy, N)
        #print("Policy = ",policy)
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor)
        #print("new_Policy = ",new_policy)
        
        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for DSC-PI-FP algorithm = {} (s) ".format(ProcessTime))
    return policy, ProcessTime, k

def discount_policy_Iteration_Csr_GJ(P, Ria, N, A, d_factor): # (PI + FP) Policy Iteration, using Gauss-Jordan eliminitation 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ DSC: PI + GJ algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        H = discount_policy_evaluation_GJ(P, Ria, d_factor, policy, N)
        #print("Policy = ",policy)
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor)
        #print("new_Policy = ",new_policy)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for DSC-PI-GJ algorithm = {} (s) ".format(ProcessTime))
    return policy, ProcessTime, k

def discount_policy_Iteration_Csr_Chiu_Rob(P, Ria, N, A, Parts, d_factor): # # (PI + Chiu+ RobB) Proposed Policy Iteration, SISDMDP structure  
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ DSC: PI + Chiu + Rob algorithm : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= MAX_ITER):
        k += 1
        print("---> Iteration k = ",k)
        H = discount_policy_evaluation_SISDMCSC(P, Ria, d_factor, policy, N, Parts)
        #print("Policy = ",policy)
        #print("H = ",H)
        new_policy = policy_Improvement_Csr(P, Ria, policy, H, N, A, d_factor)
        #print("new_Policy = ",new_policy)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for DSC-PI-SISCSDMC algorithm = {} (s) ".format(ProcessTime))
    return policy, ProcessTime, k

#---------------- B- Value Iteration Algorithm 

def discount_Value_Iteration_Csr(P, Ria, N, A, d_factor) :  # (VI) Value Iteration for "Sparse CSR" actions matrixes   
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ DSC: VI algorithm  :  Sparse Matrix version @@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
    #historique_variations = []  # Pour suivre l'historique des variations
    k = 0 

    while(k <= MAX_ITER):
        k += 1
        #J_prev = np.copy(J)
        J_prev = J.copy()

        Q = np.zeros((N, A))
        for a in range(A):
            Q[:, a] = Ria[:, a] +d_factor* P[a].dot(J_prev)
        J = np.max(Q, axis=1)

        norme = np.max(np.abs(J - J_prev))
        print("Iteration k = {}, norme = {:.15e}".format(k,norme))
        if norme < EPSILON:
            break

    Optimal_Policy = np.argmax(Q, axis=1)
    print("Iteration k = {}, norme = {:.15e}".format(k,norme))
    #print("J = ",J)
    #print("policy =",Optimal_Policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for DSC-VI algorithm = {} (s) ".format(ProcessTime))
    return Optimal_Policy, ProcessTime, k
