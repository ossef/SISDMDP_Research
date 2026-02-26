from generate_matrix import generate_sparse_matrix

from Solve_Avg_Dsc_Reward import average_relative_Value_Iteration_Csr
from Solve_Avg_Dsc_Reward import average_policy_Iteration_Csr_FP
from Solve_Avg_Dsc_Reward import average_policy_Iteration_Csr_GJ
from Solve_Avg_Dsc_Reward import average_policy_Iteration_Csr_Chiu_Rob
from Solve_Avg_Dsc_Reward import average_policy_Iteration_Csr_Chiu

from Solve_Avg_Dsc_Reward import discount_Value_Iteration_Csr
from Solve_Avg_Dsc_Reward import discount_policy_Iteration_Csr_FP
from Solve_Avg_Dsc_Reward import discount_policy_Iteration_Csr_GJ
from Solve_Avg_Dsc_Reward import discount_policy_Iteration_Csr_Chiu_Rob

from Solve_Avg_Dsc_Reward import average_Measures
from Graph import Graph

import time, os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib.backends.backend_pdf import PdfPages


def generate_Chiu_SISDMDP() :
    """ Generating the [Chiu-Feinberg,1987] MDP, adapted to SISDMC-SC structure """
 
    ''' ---------- B++ structure ---------'''
    B1 = [[0,   0.3, 0.5,  0.2, 0,   0],
        [1,   0,   0,    0,   0,   0], 
        [1, 0,   0,      0, 0,   0],
        [0.4, 0,   0,    0,   0.3, 0.3],
        [0,   0,   0,    1,   0,   0],
        [0,   0,   0,    1,   0,   0]]

    B2 = [[0, 0.3, 0.2, 0.3, 0, 0, 0.2, 0, 0],
        [1, 0,   0,   0,   0, 0, 0,   0, 0],
        [1, 0,   0,   0,   0, 0, 0,   0, 0],
        [0.2, 0, 0,   0, 0.4, 0.3,0.1, 0, 0],
        [0,   0, 0,   1,   0, 0,  0,  0, 0],
        [0, 0,   0,   1,   0, 0, 0,   0, 0],
        [0.4, 0, 0, 0.3, 0, 0, 0, 0.2, 0.1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0]]

    ''' ---------- SISCSDMC strcutre ---------'''
    #S1 : simple SISCSDMC Matrix
    S1 = [[0,   0.3, 0.5,  0.2, 0,   0],
        [1,   0,   0,    0,   0,   0], 
        [0.4, 0,   0,    0.6, 0,   0],
        [0.4, 0,   0,    0,   0.3, 0.3],
        [0.5,   0,   0,    0.5,   0,   0],
        [0,   0,   0,    1,   0,   0]]
    Parts_S1 = [[0, 1, 2], [3, 4, 5]]

    ''' ---------- SISCSDMC-SC-Rob-B strcutre ---------'''
    #S2 : [Chiu,1987] SISDMC-SC Matrix
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
    S22 = [[0.29, 0.26, 0, 0 ,0 ,0 ,0, 0.27, 0, 0.18], #loop in this state, a "non_R_state"
        [0, 0, 0.82, 0, 0, 0, 0, 0, 0.05, 0.13], 
        [0, 0, 0.31, 0, 0, 0, 0,  0.49, 0.20, 0 ],  #loop in this state, an "R_state"
        [0, 0, 0, 0.43, 0.42, 0, 0, 0, 0, 0.15],    #loop in this state, a "non_R_state"
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0.17, 0.54, 0.29, 0, 0],    #loop in this state
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0.41, 0.30, 0.29, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.25, 0.65, 0, 0, 0, 0, 0.10],
        [0, 0, 0, 0, 0, 0.5, 0.4, 0, 0, 0.1]        #loop in this state, a "non_R_state" (also superState)
        ]
    S23 = [[0.39, 0.26, 0, 0 ,0 ,0 ,0, 0.27, 0, 0.08], #loop in this state, a "non_R_state"
        [0, 0, 0.62, 0, 0, 0, 0, 0, 0.25, 0.13], 
        [0, 0, 0.31, 0, 0, 0, 0,  0.49, 0.20, 0 ],  #loop in this state, an "R_state"
        [0, 0, 0, 0.23, 0.62, 0, 0, 0, 0, 0.15],    #loop in this state, a "non_R_state"
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0.17, 0.54, 0.29, 0, 0],    #loop in this state
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0.41, 0.20, 0.39, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.25, 0.55, 0, 0, 0, 0, 0.20],
        [0, 0, 0, 0, 0, 0.3, 0.4, 0, 0, 0.3]        #loop in this state, a "non_R_state" (also superState)
        ]

    '''------ Construction d'un MDP avec 3 actions  -------------'''
    Parts_S2 = [[7, 0, 1, 2], [9, 5, 6], [8, 3, 4]]                           #Les partitions
    S2_sparse = np.array([csr_matrix(S21), csr_matrix(S22), csr_matrix(S23)]) #Une matrice de taille N=10 (en sparse) par action
    N, A = 10, 3
    R_S2 = np.random.randint(1, N, size=(N, A))

    return S2_sparse, R_S2, N, A, Parts_S2

def generate_synthetic_SISDMDP(k, N, A, epsilon=1e-3):
    """ Generating the synthetic SISDMDP
    k : number of partitions
    N : number of states
    A : number of actions
    """

    start_time = time.time()

    #First original SISDMC-SC Markov chain generation
    P_base, superstates = generate_sparse_matrix(k, N, epsilon= 1e-3)
    N = P_base.shape[0]
    matrices = []
    matrices.append(P_base)

    #Access CSR
    base_data = P_base.data
    base_indices = P_base.indices
    base_indptr = P_base.indptr

    #Other Actions generation
    for _ in range(A-1):
        data_new = []
        indices_new = []
        indptr_new = [0]

        for i in range(N):
            start = base_indptr[i]
            end = base_indptr[i+1]
            cols = base_indices[start:end]
            n = len(cols)

            if n == 0:
                indptr_new.append(len(data_new))
                continue

            # Nouveau tirage de proba aléatoires positives
            tirages = np.random.rand(n)
            tirages /= tirages.sum()

            data_new.extend(tirages)
            indices_new.extend(cols)
            indptr_new.append(len(data_new))

        matrices.append(csr_matrix((data_new, indices_new, indptr_new), shape=(N, N)))

    S_sparse = np.array(matrices)

    #Rewards generation
    R = np.random.normal(loc=50, scale=15, size=(N, A))
    
    end_time = time.time()
    print(f"\n✅ SISDMDP Generation time :  {end_time - start_time:.4f} sec")

    return S_sparse, R, N, A, superstates

def generate_reward_fast0(N, A, ALL_Ps, states, r1_by_M, r2, r3,BufferSize, n_packets, h_deb, pService):
    # Initialiser la matrice de récompense
    print("Generate rewards fast 0")

    states = np.asarray(states)
    m = states[:, 0].astype(int)
    d = states[:, 1].astype(int)
    h = states[:, 2].astype(int)
    x = states[:, 3].astype(int)

    mask_release = (d == x) & (h == h_deb[m])  # Release (energy selling) states
    mask_full    = (x == BufferSize)           # Energy loss states (full buffer)

    idx_release = np.where(mask_release)[0]
    idx_full    = np.where(mask_full)[0]

    # ---- QoS vector on next-states: qos_next[j] = P(service requested | s'=j) if X_j==0 else 0
    qos_next = np.zeros(N, dtype=float)
    for j in range(N):
        if x[j] == 0:
            hh = h[j] - h_deb[m[j]]
            qos_next[j] = pService[hh]

    # overflow coeff by x1
    overflow_coeff = np.zeros(BufferSize + 1, dtype=float)
    for x1 in range(BufferSize + 1):
        s = 0.0
        for e in range(n_packets):
            for b in range(2):
                s += max(0, x1 + e - b - BufferSize)
        overflow_coeff[x1] = s

    reward = np.zeros((N, A), dtype=float)
    x_release = x  # x2 vector

    # Sécuriser le type: r1_by_M peut être list/np.array ou dict
    if isinstance(r1_by_M, dict):
        r1_vec = np.array([r1_by_M[int(mi)] for mi in m], dtype=float)
    else:
        r1_by_M = np.asarray(r1_by_M, dtype=float)
        r1_vec = r1_by_M[m]  # r1(M) par état

    for a in range(A):
        P = ALL_Ps[a]  # CSR

        # ---- QoS delay: E[ 1_{X'=0} * pService(h') | s, a ]
        E_qos = np.asarray(P.dot(qos_next)).ravel()
        reward[:, a] += r3 * E_qos  # r3 < 0 (penalty)

        # full (overflow proxy)
        p_full = np.asarray(P[:, idx_full].sum(axis=1)).ravel()
        reward[:, a] += r2 * overflow_coeff[x] * p_full  # r2 < 0 (penalty)

        # release (energy sold) with weather-dependent value r1(M)
        p_rel = np.asarray(P[:, idx_release].sum(axis=1)).ravel()
        ex2_rel = np.asarray(P[:, idx_release].dot(x_release[idx_release])).ravel()
        sold_expect = x * p_rel - ex2_rel
        reward[:, a] += r1_vec * sold_expect

    return reward

def generate_reward_fast(N, A, ALL_Ps, states, r1, r2, r3, BufferSize, n_packets, h_deb, pService):
    # Initialiser la matrice de récompense
    print("Generate rewards fast")

    states = np.asarray(states)
    m = states[:, 0].astype(int)
    d = states[:, 1].astype(int)
    h = states[:, 2].astype(int)
    x = states[:, 3].astype(int)

    mask_release = (d == x) & (h == h_deb[m])  # Release (energy selling) states
    mask_full    = (x == BufferSize)           # Energy loss states (full buffer)

    idx_release = np.where(mask_release)[0]
    idx_full    = np.where(mask_full)[0]

    # ---- QoS vector on next-states: qos_next[j] = P(service requested | s'=j) if X_j==0 else 0
    qos_next = np.zeros(N, dtype=float)
    for j in range(N):
        if x[j] == 0:
            hh = h[j] - h_deb[m[j]]
            qos_next[j] = pService[hh]

    # overflow coeff by x1
    overflow_coeff = np.zeros(BufferSize + 1, dtype=float)
    for x1 in range(BufferSize + 1):
        s = 0.0
        for e in range(n_packets):
            for b in range(2):
                s += max(0, x1 + e - b - BufferSize)
        overflow_coeff[x1] = s

    reward = np.zeros((N, A), dtype=float)

    x_release = x  # x2 vector

    for a in range(A):
        P = ALL_Ps[a]  # CSR

        # ---- QoS delay: E[ 1_{X'=0} * pService(h') | s, a ]
        E_qos = np.asarray(P.dot(qos_next)).ravel()
        reward[:, a] += r3 * E_qos  # r3 < 0 (penalty)

        # full (overflow proxy)
        p_full = np.asarray(P[:, idx_full].sum(axis=1)).ravel()
        reward[:, a] += r2 * overflow_coeff[x] * p_full  # r2 < 0 (penalty)

        # release (energy sold)
        p_rel = np.asarray(P[:, idx_release].sum(axis=1)).ravel()
        ex2_rel = np.asarray(P[:, idx_release].dot(x_release[idx_release])).ravel()
        sold_expect = x * p_rel - ex2_rel
        reward[:, a] += r1 * sold_expect  # r1 > 0

    return reward

def generate_reward_slow(N, A, ALL_Ps, states, r1, r2, r3, BufferSize, n_packets, h_deb, pService):
    # Initialiser la matrice de récompense
    reward_matrix = np.zeros((N, A))
    print("Generate rewards slow")

    for id1, etat1 in enumerate(states):
        m1, d1, h1, x1 = etat1[0], etat1[1], etat1[2], etat1[3]
        for a in range(A):
            s = 0
            for id2, etat2 in enumerate(states):
                m2, d2, h2, x2 = etat2[0], etat2[1], etat2[2], etat2[3]

                #Reward, for battery release 
                #if ( (d2 == x2) and (d2 in release_vals) and (h2 == h_deb[m2]) ) :
                if ( (d2 == x2) and (h2 == h_deb[m2]) ) :
                    s += ALL_Ps[a][id1,id2]*(x1-x2)*r1
                    #print(f'Action {a}: Release from ({m1}, {d1}, {h1}, {x1}) --> ({m2}, {d2}, {h2}, {x2})')

                #Penalty for packets loss
                if(x2 == BufferSize):       
                    r = 0
                    for e in range(n_packets):
                        for b in range(2) :
                            r += ALL_Ps[a][id1,id2]*max(0,x1+e-b-BufferSize)*r2
                    s+=r

                #Penalty for empty battery (i.e. packets delay): going to empty states
                if (x2 == 0) :
                    s += ALL_Ps[a][id1,id2]*pService[h2 - h_deb[m2]]*r3
                    #print(f'Empty: ({m1}, {d1}, {h1}, {x1}) --> ({m2}, {d2}, {h2}, {x2})')
            reward_matrix[id1][a] = s
    return reward_matrix

def load_weather_and_service(city):
    """
    Returns:
      weather_hours_packets : list length 4, each mat shape (T_m, num_paquets)
      h_deb                : np.ndarray (4,)
      h_fin                : np.ndarray (4,)
      num_paquets          : int  (#packet values, packets 0..num_paquets-1)
      packet_size          : int  (Wh)
      pService_hourly      : np.ndarray (24,) with pService_hourly[h] = P(service at hour h)
    """
    path = "../Create_Model/Model_Release_Meteo/SIMPA_Journal_Model/"
    weather_files = [
        path+"NREL_Extracts/"+city+"/"+city+"_M0.data",
        path+"NREL_Extracts/"+city+"/"+city+"_M1.data",
        path+"NREL_Extracts/"+city+"/"+city+"_M2.data",
        path+"NREL_Extracts/"+city+"/"+city+"_M3.data",
    ]
    service_file  = path+"NREL_Extracts/Service_Demand.data"
    actions_file  = path+"NREL_Model/Actions.txt"

    # --------- Weather/Hour/Arrivals ------
    weather_hours_packets = []
    h_deb, h_fin = [], []
    num_paquets = None
    packet_size = None

    for wf in weather_files:
        with open(wf, "r", encoding="utf-8", errors="replace") as f:
            f.readline()  # title
            meta = f.readline().strip().replace("\t", " ").split()
            hs, hf, k, ps = int(meta[0]), int(meta[1]), int(meta[2]), int(float(meta[3]))
            f.readline()  # header line

            if num_paquets is None:
                num_paquets = k
                packet_size = ps

            T = hf - hs + 1
            mat = np.zeros((T, num_paquets), dtype=float)

            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().replace("\t", " ").split()
                hour = int(parts[0])
                mat[hour - hs, :] = np.array(parts[1:1+num_paquets], dtype=float)

        weather_hours_packets.append(mat)
        h_deb.append(hs)
        h_fin.append(hf)

    h_deb = np.array(h_deb, dtype=int)
    h_fin = np.array(h_fin, dtype=int)

    # --------- Service (hourly lookup) ------
    pService_hourly = np.zeros(24, dtype=float)
    with open(service_file, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # first line: hour start/end (not needed further)
        for line in f:
            if not line.strip():
                continue
            hh, pp = line.strip().replace("\t", " ").split()[:2]
            pService_hourly[int(hh)] = float(pp)
    
    # ------- Actions distribution ----
    with open(actions_file, "r", encoding="utf-8", errors="replace") as f:
        A = int(f.readline().strip())      # number of actions
        C = int(f.readline().strip())      # number of components per action (here 4)
        pSellAction = np.zeros((A, C), dtype=float) # depends on action

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().replace("\t", " ").split()
            a = int(parts[0])
            pSellAction[a, :] = np.array([float(x) for x in parts[1:1 + C]], dtype=float)

    return weather_hours_packets, h_deb, h_fin, num_paquets, packet_size, pService_hourly, pSellAction

def generate_XBorne_SISDMDP(city, r1_by_M, r2, r3, BufferSize):
    """ 
    Read Xborne Models: state (M, D, H, X)
    Each action "i" is represented by files: Model_ai.sz, Model_ai.Rii, Model_ai.cd, Model_ai.part
    Partition does not change with action 
    """

    #----- 1 - reading the number of Actions 
    Model = "../Create_Model/Model_Release_Meteo/SIMPA_Journal_Model/NREL_Model/"
    with open(Model+"Actions.txt", 'r') as file:
        lines = file.readlines()
        A , K = int(lines[0]), int(lines[1])

    print("=>  Reading all sparse matrixes")
    print(f"=> A = {A} actions of {K} elements")
    start_time = time.time()

    #----- 2 - reading only once: N, BufferSize, Deadline, Parts
    myGraph     = Graph(Model,city)  #Initalize and read a ".sz" file of action 
    N           = myGraph.N 
    superstates = myGraph.superstates

    #----- 3 - read TPM(a) : Matrixes model from external file for --------
    All_Ps = []
    for a in range(A) :
        myGraph.read_Rii_Matrixe(a)
        Ps = myGraph.csr_sparse
        All_Ps.append(Ps)
    #All_Ps = np.array(All_Ps)
    states = myGraph.states

    #Load some "NREL_Extracts" parameters needed for "average measure function"
    weather_hours_packets, h_deb, h_fin, n_packets, packet_size, pService_hourly, pSellAction = load_weather_and_service(city)

    #Rewards generation
    R = generate_reward_fast0(
        N=N,
        A=A,
        ALL_Ps=All_Ps,
        states=states,
        r1_by_M=r1_by_M,
        r2=r2,
        r3=r3,
        BufferSize=BufferSize,
        n_packets=n_packets,
        h_deb=h_deb,
        pService=pService_hourly,
    )

    #R = generate_reward_fast2(N, A, All_Ps, states, r1_by_M, r2, r3, BufferSize, n_packets, h_deb, pService_hourly,pSellAction, lambda_peak=0.02, lambda_sell2=0)
    #R = generate_reward_slow(N, A, All_Ps, states, r1, r2, r3, BufferSize, n_packets, h_deb)

    ProcessTime = time.time() - start_time
    print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

    return All_Ps, states, R, N, A, superstates, weather_hours_packets, h_deb, h_fin, n_packets, packet_size, pService_hourly, pSellAction 

def test_scalability_algorithms() :
    """------------1) SISDMDP Generation -----------------------"""
    K, N, A = 10, 1000, 200
    ALL_P, R, N, A, Parts = generate_synthetic_SISDMDP(K, N, A)

    """----------- 2) Solving Average reward criteria ----------"""
    relative_state = Parts[0][0]  #Only for Average reward !
    rho_RB, policy_RRB, time_RRB, k_RRB  = average_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N, A, Parts, relative_state)
    rho_RC, policy_RC, time_RC, k_RC     = average_policy_Iteration_Csr_Chiu(ALL_P, R, N, A, Parts, relative_state)
    rho_GJ, policy_RGJ, time_RGJ, k_RGJ  = average_policy_Iteration_Csr_GJ(ALL_P, R, N, A, relative_state)
    rho_FP, policy_RFP, time_RFP, k_RFP  = average_policy_Iteration_Csr_FP(ALL_P, R, N, A, relative_state)
    rho_RVI, policy_RVI, time_RVI, k_RVI = average_relative_Value_Iteration_Csr(ALL_P, R, N, A, relative_state)

    print(f'\nAVG Results : K={len(Parts)}, N={N}, A={A}')
    print(f'--> RPI_RB  : rau = {rho_RB} , time = {time_RRB}, iter = {k_RRB}')
    print(f'--> RPI_RC  : rau = {rho_RC} , time = {time_RC}, iter = {k_RC}')
    print(f'--> RPI_GJ  : rau = {rho_GJ} , time = {time_RGJ}, iter = {k_RGJ}')
    print(f'--> RPI_FP  : rau = {rho_FP} , time = {time_RFP}, iter = {k_RFP}')
    print(f'--> RVI     : rau = {rho_RVI} , time = {time_RVI}, iter = {k_RVI}')

    vals = np.array([rho_RB, rho_RVI, rho_RC, rho_FP, rho_GJ])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e  for a,b,c,d,e in zip(policy_RRB,policy_RVI, policy_RC,policy_RGJ, policy_RFP)) ):
        print("--------> Policy Similarity OK !")
    else :
        print("--------> Policy Similarity KO !")


    """---------- 3) Solving Discounted reward criteria ----------"""
    """
    d_factor = 0.9        #Only for Discounted reward !
    policy_RB, time_RB, k_RB = discount_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N, A, Parts, d_factor)
    policy_VI, time_VI, k_VI = discount_Value_Iteration_Csr(ALL_P, R, N, A, d_factor)
    policy_FP, time_FP, k_FP = discount_policy_Iteration_Csr_FP(ALL_P, R, N, A, d_factor)
    policy_GJ, time_GJ, k_GJ = discount_policy_Iteration_Csr_GJ(ALL_P, R, N, A, d_factor)

    print(f'\nDSC Results : K={K}, N={N}, A={A}')
    print(f'--> VI     : time = {time_VI}, iter = {k_VI}')
    print(f'--> PI_FP  : time = {time_FP}, iter = {k_FP}')
    print(f'--> PI_GJ  : time = {time_GJ}, iter = {k_GJ}')
    print(f'--> PI_RB  : time = {time_RB}, iter = {k_RB}')

    if( all(a == b == c == d   for a,b,c,d in zip(policy_FP,policy_GJ,policy_RB,policy_VI)) ):
        print("--------> Policy Similarity OK !")
    else :
        print("--------> Policy Similarity KO !")
    """

def test_scalability_algorithms_CI95(n_runs):
    """
    -Computes the 95% confidence interval of execution time width (as a percentage of the mean)
    n_runs: number of repeated experiments (with different seeds)
    -Normalizes the execution time of PI+RB, PI+FP, and PI+GJ
    to the maximum number of outer iterations observed (shared across them),
    """
    K, N, A = 10, 5000, 200
    d_factor = 0.9    #Only for Discounted reward !

    times = {
        "PI+RB": [],
        "PI+FP": [],
        "PI+GJ": [],
        #"PI+RC": [],  #Only for Average reward !
        "VI": []
    }
    shared_iters = []  # same k for PI+*

    for seed in range(n_runs):
        np.random.seed(seed)
        ALL_P, R, N_gen, A_gen, Parts = generate_synthetic_SISDMDP(K, N, A)
        print(f'\n==> Experiment {seed}/{n_runs}')

        """---------- Solving Average reward criteria ----------"""
        #relative_state = Parts[0][0]  #Only for Average reward !
        #_, _, time_rb, k = average_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N, A, Parts, relative_state)
        #_, _, time_rc, _ = average_policy_Iteration_Csr_Chiu(ALL_P, R, N, A, Parts, relative_state)
        #_, _, time_fp, _ = average_policy_Iteration_Csr_FP(ALL_P, R, N, A, relative_state)
        #_, _, time_gj, _ = average_policy_Iteration_Csr_GJ(ALL_P, R, N, A, relative_state)
        #_, _, time_vi, _ = average_relative_Value_Iteration_Csr(ALL_P, R, N, A, relative_state)

        #shared_iters.append(k)
        #times["PI+RB"].append(time_rb)
        #times["PI+FP"].append(time_fp)
        #times["PI+GJ"].append(time_gj)
        #times["PI+RC"].append(time_rc)
        #times["VI"].append(time_vi)

        """---------- Solving Discounted reward criteria ----------"""
        _, time_rb, k = discount_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N_gen, A_gen, Parts, d_factor)
        _, time_fp, _ = discount_policy_Iteration_Csr_FP(ALL_P, R, N_gen, A_gen, d_factor)
        _, time_gj, _ = discount_policy_Iteration_Csr_GJ(ALL_P, R, N_gen, A_gen, d_factor)
        _, time_vi, _ = discount_Value_Iteration_Csr(ALL_P, R, N_gen, A_gen, d_factor)

        shared_iters.append(k)
        times["PI+RB"].append(time_rb)
        times["PI+FP"].append(time_fp)
        times["PI+GJ"].append(time_gj)
        times["VI"].append(time_vi)

    max_k = max(shared_iters)

    # Normalize PI+* times to max_k
    for key in ["PI+RB", "PI+FP", "PI+GJ" ]: #, "PI+RC"]:
        norm_times = [
            (t / k) * max_k for t, k in zip(times[key], shared_iters)
        ]
        times[key] = norm_times

    print(f'\nNorm DSC Results : K={K}, N={N}, A={A}')
    # Compute and print confidence intervals
    for algo_name, t_list in times.items():
        mean = np.mean(t_list)
        sem = stats.sem(t_list)
        ci_low, ci_high = stats.t.interval(0.95, len(t_list) - 1, loc=mean, scale=sem)
        ci_width_percent = 100 * (ci_high - ci_low) / mean

        print(f"{algo_name:7s}: Mean time = {mean:.4f} s | 95% CI = ±{ci_width_percent:.2f}%" + (f" (k={max_k})" if "PI" in algo_name else ""))

def plot_policy_heatmaps(
    states,
    policy_idx,
    h_deb,
    h_fin,
    city,
    r1_by_M,r2,r3,
    BufferSize=None,
    regimes_names=None,
    show=False,
    dpi=150
):
    """
    Generate and save policy heatmaps (X,H)
    for each (M,D) root in output_dir/.
    """

    output_dir ="HeatMaps/"+city+"_r2_"+str(r2)+"_r3_"+str(r3)
    os.makedirs(output_dir, exist_ok=True)

    def unpack_state(s):
        return s[0], s[1], s[2], s[3]  # (M, D, H, X)

    def get_a(i, s):
        return policy_idx[i]
    
    n_actions = max(policy_idx) + 1

    # BufferSize
    if BufferSize is None:
        BufferSize = int(max(unpack_state(s)[3] for s in states))

    def name_M(M):
        if regimes_names is None:
            return f"M={M}"
        if isinstance(regimes_names, dict):
            return regimes_names.get(M, f"M={M}")
        return regimes_names[M]

    def get_hdeb(M): return h_deb[M] if not isinstance(h_deb, dict) else h_deb.get(M)
    def get_hfin(M): return h_fin[M] if not isinstance(h_fin, dict) else h_fin.get(M)

    # group by (M, D)
    by_root = defaultdict(list)
    for i, s in enumerate(states):
        M, D, H, X = unpack_state(s)
        by_root[(M, D)].append((i, H, X, s))

    for (M, D), items in sorted(by_root.items()):
        hd = get_hdeb(M)
        hf = get_hfin(M)

        H_values = list(range(int(hd), int(hf) + 1))
        H_to_col = {H: c for c, H in enumerate(H_values)}

        # 🔄 MATRICE TRANSPOSEE : (X, H)
        mat = np.full((BufferSize + 1, len(H_values)), np.nan)

        for (i, H, X, s) in items:
            a = get_a(i, s)
            if a is None:
                continue
            if 0 <= X <= BufferSize and H in H_to_col:
                mat[X, H_to_col[H]] = a

        fig, ax = plt.subplots(figsize=(9, 4.5))
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            vmin=-0.5,
            vmax=n_actions - 0.5
        )

        # 🔤 TITRES & LABELS EN ANGLAIS
        ax.set_title(f"{name_M(M)} — D={D}")
        ax.set_xlabel("H (Hour)")
        ax.set_ylabel("X (Energy packets)")

        # X-axis ticks (Hours)
        step = 1 if len(H_values) <= 16 else max(1, len(H_values)//10)
        xticks = list(range(0, len(H_values), step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(H_values[c]) for c in xticks])

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Action")

        # afficher seulement quelques ticks lisibles
        max_labels = 10
        ticks = np.linspace(0, n_actions - 1, max_labels, dtype=int)

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])


        fig.tight_layout()

        filename = f"{city}_{name_M(M).replace(' ', '_')}_D{D}.png"
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_policy_decision_maps_by_M(
    states,
    policy_idx,
    h_deb,
    h_fin,
    city,
    r2,
    r3,
    BufferSize=None,
    D_max=None,              # par défaut: 3/4*BufferSize
    regimes_names=None,
    show=False,
    dpi=150
):
    """
    One figure per M, only decision states:
      - Subplot 1: End-of-day decision states (H == h_fin[M]) as heatmap X vs D
      - Subplot 2: Full-buffer decision states (X == BufferSize) as heatmap D vs H
    """

    output_dir = f"HeatMaps/{city}_r2_{r2}_r3_{r3}_DecisionOnly"
    os.makedirs(output_dir, exist_ok=True)

    # numpy view of states
    states = np.asarray(states)
    m = states[:, 0].astype(int)
    d = states[:, 1].astype(int)
    h = states[:, 2].astype(int)
    x = states[:, 3].astype(int)

    N = len(states)
    if BufferSize is None:
        BufferSize = int(x.max())

    if D_max is None:
        D_max = BufferSize

    n_actions = int(np.max(policy_idx)) + 1 if len(policy_idx) else 1

    def name_M(M):
        if regimes_names is None:
            return f"M={M}"
        if isinstance(regimes_names, dict):
            return regimes_names.get(M, f"M={M}")
        return regimes_names[M]

    def get_hdeb(M): return h_deb[M] if not isinstance(h_deb, dict) else h_deb.get(M)
    def get_hfin(M): return h_fin[M] if not isinstance(h_fin, dict) else h_fin.get(M)

    Ms = sorted(set(m.tolist()))

    # ticks for colorbar (avoid 66 labels)
    cbar_ticks = np.linspace(0, n_actions - 1, 10, dtype=int) if n_actions > 1 else [0]

    for M in Ms:
        hd = int(get_hdeb(M))
        hf = int(get_hfin(M))

        # --------------------------
        # Subplot 1: End-of-day H=hf
        # --------------------------
        sel_eod = (m == M) & (h == hf) & (d >= 0) & (d <= D_max) & (x >= 0) & (x <= BufferSize)
        idx_eod = np.where(sel_eod)[0]

        # Heatmap axes:
        #   x-axis: D in [0..D_max]
        #   y-axis: X in [0..BufferSize]
        mat_eod = np.full((BufferSize + 1, D_max + 1), np.nan)
        for i in idx_eod:
            D = int(d[i]); X = int(x[i])
            mat_eod[X, D] = policy_idx[i]

        # -------------------------------
        # Subplot 2: Full buffer X=Buffer
        # -------------------------------
        sel_full = (m == M) & (x == BufferSize) & (d >= 0) & (d <= D_max) & (h >= hd) & (h <= hf)
        idx_full = np.where(sel_full)[0]

        H_values = list(range(hd, hf + 1))
        H_to_col = {H: j for j, H in enumerate(H_values)}

        # Heatmap axes:
        #   x-axis: H in [hd..hf]
        #   y-axis: D in [0..D_max]
        mat_full = np.full((D_max + 1, len(H_values)), np.nan)
        for i in idx_full:
            D = int(d[i]); H = int(h[i])
            mat_full[D, H_to_col[H]] = policy_idx[i]

        # ---- Plot figure (1 per M) ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # --- EOD plot ---
        im1 = ax1.imshow(
            mat_eod,
            origin="lower",
            aspect="auto",
            vmin=-0.5, vmax=n_actions - 0.5
        )
        ax1.set_title(f"{city} — {name_M(M)} — Decision at end of day (H={hf})")
        ax1.set_xlabel("D")
        ax1.set_ylabel("X (Energy packets)")

        # ticks for D
        if D_max <= 20:
            ax1.set_xticks(range(0, D_max + 1, 1))
        else:
            stepD = max(1, (D_max + 1) // 10)
            ax1.set_xticks(list(range(0, D_max + 1, stepD)))

        # ticks for X
        if BufferSize <= 20:
            ax1.set_yticks(range(0, BufferSize + 1, 1))
        else:
            stepX = max(1, (BufferSize + 1) // 10)
            ax1.set_yticks(list(range(0, BufferSize + 1, stepX)))

        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label("Action")
        cbar1.set_ticks(cbar_ticks)
        cbar1.set_ticklabels([str(t) for t in cbar_ticks])

        # --- FULL plot ---
        im2 = ax2.imshow(
            mat_full,
            origin="lower",
            aspect="auto",
            vmin=-0.5, vmax=n_actions - 0.5
        )
        ax2.set_title(f"{city} — {name_M(M)} — Decision when buffer is full (X={BufferSize})")
        ax2.set_xlabel("H (Hour)")
        ax2.set_ylabel("D")

        # ticks for H
        if len(H_values) <= 16:
            ax2.set_xticks(range(len(H_values)))
            ax2.set_xticklabels([str(H) for H in H_values], rotation=45, ha="right")
        else:
            stepH = max(1, len(H_values) // 10)
            xt = list(range(0, len(H_values), stepH))
            ax2.set_xticks(xt)
            ax2.set_xticklabels([str(H_values[j]) for j in xt], rotation=45, ha="right")

        # ticks for D
        if D_max <= 20:
            ax2.set_yticks(range(0, D_max + 1, 1))
        else:
            stepD2 = max(1, (D_max + 1) // 10)
            ax2.set_yticks(list(range(0, D_max + 1, stepD2)))

        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label("Action")
        cbar2.set_ticks(cbar_ticks)
        cbar2.set_ticklabels([str(t) for t in cbar_ticks])

        # Save
        filename = f"{city}_{name_M(M).replace(' ', '_')}_DecisionOnly.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_policy_decision_maps_by_M_pdf_2x2(
    states,
    policy_idx,
    h_deb,
    h_fin,
    city,
    r2,
    r3,
    BufferSize=None,
    D_max=None,              # par défaut: BufferSize
    regimes_names=None,
    show=False,
    dpi=150
):
    """
    Une seule page PDF avec 4 heatmaps (2x2), une par régime météo M.
    Chaque heatmap = états de décision en fin de journée (H == h_fin[M]) : X vs D.
    """

    output_dir = f"HeatMaps/B{BufferSize}_{city}_r2_{r2}_r3_{r3}_DecisionOnly"
    os.makedirs(output_dir, exist_ok=True)

    # numpy view of states
    states = np.asarray(states)
    m = states[:, 0].astype(int)
    d = states[:, 1].astype(int)
    h = states[:, 2].astype(int)
    x = states[:, 3].astype(int)

    if BufferSize is None:
        BufferSize = int(x.max())

    if D_max is None:
        D_max = BufferSize

    n_actions = int(np.max(policy_idx)) + 1 if len(policy_idx) else 1

    def name_M(M):
        if regimes_names is None:
            return f"M={M}"
        if isinstance(regimes_names, dict):
            return regimes_names.get(M, f"M={M}")
        return regimes_names[M]

    def get_hdeb(M): return h_deb[M] if not isinstance(h_deb, dict) else h_deb.get(M)
    def get_hfin(M): return h_fin[M] if not isinstance(h_fin, dict) else h_fin.get(M)

    Ms = sorted(set(m.tolist()))

    # On s'attend à 4 régimes, mais on gère proprement si != 4
    n = len(Ms)
    nrows, ncols = 2, 2
    max_plots = nrows * ncols
    Ms = Ms[:max_plots]

    # ticks for colorbar (évite 66 labels)
    cbar_ticks = np.linspace(0, n_actions - 1, 10, dtype=int) if n_actions > 1 else [0]

    pdf_path = os.path.join(output_dir, f"B{BufferSize}_{city}_r2_{r2}_r3_{r3}_DecisionOnly_2x2.pdf")

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        last_im = None

        for k, M in enumerate(Ms):
            ax = axes[k]
            hf = int(get_hfin(M))

            # --- sélection fin de journée H = hf ---
            sel_eod = (
                (m == M) &
                (h == hf) &
                (d >= 0) & (d <= D_max) &
                (x >= 0) & (x <= BufferSize)
            )
            idx_eod = np.where(sel_eod)[0]

            # Heatmap axes:
            #   x-axis: D in [0..D_max]
            #   y-axis: X in [0..BufferSize]
            mat_eod = np.full((BufferSize + 1, D_max + 1), np.nan)
            for i in idx_eod:
                D = int(d[i]); X = int(x[i])
                mat_eod[X, D] = policy_idx[i]

            last_im = ax.imshow(
                mat_eod,
                origin="lower",
                aspect="auto",
                vmin=-0.5, vmax=n_actions - 0.5
            )

            ax.set_title(f"{city} — {name_M(M)} — Décision fin de journée (H={hf})")
            ax.set_xlabel("D")
            ax.set_ylabel("X (Energy packets)")

            # ticks pour D
            if D_max <= 20:
                ax.set_xticks(range(0, D_max + 1, 1))
            else:
                stepD = max(1, (D_max + 1) // 10)
                ax.set_xticks(list(range(0, D_max + 1, stepD)))

            # ticks pour X
            if BufferSize <= 20:
                ax.set_yticks(range(0, BufferSize + 1, 1))
            else:
                stepX = max(1, (BufferSize + 1) // 10)
                ax.set_yticks(list(range(0, BufferSize + 1, stepX)))

        # Si moins de 4 régimes, on masque les axes restants
        for k in range(len(Ms), max_plots):
            axes[k].axis("off")

        # Colorbar commune (à droite)
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.9)
            cbar.set_label("Action")
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([str(t) for t in cbar_ticks])

        # Sauvegarde 1 page PDF
        pdf.savefig(fig, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_policy_threshold_summary_by_M_pdf(
    states,
    policy_idx,
    h_fin,
    city,
    r2,
    r3,
    pSellAction,              # (A,4) : [p_sell_all, p_sell_2/3, p_sell_1/3, p_conserve]
    BufferSize=None,
    D_max=None,
    regimes_names=None,
    avg_release=None,
    avg_loss=None,
    qos_delay_prob=None,
    show=False,
    dpi=150
):
    """
    1 figure PDF: for each regime M, one vertical segmented bar indexed by X (0..BufferSize)
    at the release time h == h_fin[M]. Colors indicate the dominant selling option.

    Dominant action is obtained via argmax(pSellAction[a]).
    """

    # -----------------
    # Sanity checks
    # -----------------
    pSellAction = np.asarray(pSellAction, dtype=float)
    if pSellAction.ndim != 2 or pSellAction.shape[1] != 4:
        raise ValueError(f"pSellAction must be of shape (A,4). Got {pSellAction.shape}")

    states = np.asarray(states)
    m = states[:, 0].astype(int)
    d = states[:, 1].astype(int)
    h = states[:, 2].astype(int)
    x = states[:, 3].astype(int)

    policy_idx = np.asarray(policy_idx, dtype=int)
    A = pSellAction.shape[0]
    if policy_idx.min() < 0 or policy_idx.max() >= A:
        raise ValueError(f"policy_idx out of range [0, {A-1}]")

    if BufferSize is None:
        BufferSize = int(x.max())
    if D_max is None:
        D_max = BufferSize

    output_dir = f"HeatMaps/B{BufferSize}_{city}_r2_{r2}_r3_{r3}_DecisionOnly"
    os.makedirs(output_dir, exist_ok=True)

    def name_M(M):
        if regimes_names is None:
            return str(M)
        if isinstance(regimes_names, dict):
            return regimes_names.get(M, str(M))
        return regimes_names[M]

    def get_hfin(M):
        return h_fin[M] if not isinstance(h_fin, dict) else h_fin.get(M)

    Ms = sorted(set(m.tolist()))[:4]

    # -------------------------------
    # Build dominant action per (M, X) at release states
    # -------------------------------
    action_map = {M: np.full(BufferSize + 1, -1, dtype=int) for M in Ms}

    for M in Ms:
        hf = int(get_hfin(M))
        sel = (m == M) & (h == hf) & (d >= 0) & (d <= D_max) & (x >= 0) & (x <= BufferSize)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue

        for X in range(BufferSize + 1):
            idx_X = idx[x[idx] == X]
            if len(idx_X) == 0:
                continue
            a_mode = Counter(policy_idx[idx_X]).most_common(1)[0][0]
            action_map[M][X] = int(a_mode)

    used_actions = sorted({int(a) for M in Ms for a in action_map[M] if a >= 0})
    if not used_actions:
        raise ValueError("No release-state data found.")

    # -------------------------------
    # Map action -> dominant selling type
    # -------------------------------
    label_by_choice = {
        0: "Sell All",
        1: "Sell 2X/3",
        2: "Sell X/3",
        3: "Conserve"
    }

    action_label = {}
    for a in used_actions:
        k = int(np.argmax(pSellAction[a]))
        action_label[a] = label_by_choice[k]

    # -------------------------------
    # Color scheme (publication-friendly)
    # -------------------------------
    TYPE_COLORS = {
        "Sell All":   "#d62728",
        "Sell 2X/3":  "#ff7f0e",
        "Sell X/3":   "#1f77b4",
        "Conserve":   "#17becf",
    }

    pdf_path = os.path.join(
        output_dir,
        f"B{BufferSize}_{city}_r2_{r2}_r3_{r3}_Summary.pdf"
    )

    with PdfPages(pdf_path) as pdf:

        # ---- Figure + font sizes ----
        fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)

        FS_AXIS = 15
        FS_TICKS = 15
        FS_LEGEND = 15
        FS_LEGEND_TITLE = 15
        FS_METRICS = 15

        bar_width = 0.35
        xs = np.arange(len(Ms))

        # -------------------------------
        # Draw segmented bars
        # -------------------------------
        for j, M in enumerate(Ms):
            a_by_X = action_map[M]

            def flush_segment(s, e, a_id):
                if a_id < 0:
                    return
                height = e - s + 1
                bottom = s
                lbl = action_label.get(a_id, "Unknown")
                color = TYPE_COLORS.get(lbl, "#7f7f7f")
                ax.bar(
                    xs[j], height,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.25
                )

            start = 0
            current_a = int(a_by_X[0])
            for X in range(1, BufferSize + 1):
                a_id = int(a_by_X[X])
                if a_id != current_a:
                    flush_segment(start, X - 1, current_a)
                    start = X
                    current_a = a_id
            flush_segment(start, BufferSize, current_a)

        # -------------------------------
        # Axes
        # -------------------------------
        ax.set_xticks(xs)
        ax.set_xticklabels([name_M(M) for M in Ms], fontsize=FS_TICKS)
        ax.set_xlabel("W (Weather regime)", fontsize=FS_AXIS)
        ax.set_ylabel("X (Energy packets)", fontsize=FS_AXIS)
        ax.set_ylim(0, BufferSize + 1)

        ax.tick_params(axis="x", labelsize=FS_TICKS)
        ax.tick_params(axis="y", labelsize=FS_TICKS)
        ax.grid(axis="y", alpha=0.25)

        # -------------------------------
        # Legend
        # -------------------------------
        labels_used = []
        for a in used_actions:
            lbl = action_label[a]
            if lbl not in labels_used:
                labels_used.append(lbl)

        handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLORS[lbl]) for lbl in labels_used]
        legend = ax.legend(
            handles, labels_used,
            title="Dominant action",
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=FS_LEGEND,
            title_fontsize=FS_LEGEND_TITLE,
            frameon=True
        )
        ax.add_artist(legend)

        # -------------------------------
        # Metrics text
        # -------------------------------
        metrics_lines = []
        if avg_release is not None:
            metrics_lines.append(f"E[Release]: {avg_release:.4g} Wh")
        if avg_loss is not None:
            metrics_lines.append(f"E[Loss]: {avg_loss:.3g} Wh")
        if qos_delay_prob is not None:
            metrics_lines.append(f"P(Delay): {qos_delay_prob:.3g}")

        if metrics_lines:
            ax.text(
                1.01, 0.72,
                "\n".join(metrics_lines),
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=FS_METRICS
            )

        pdf.savefig(fig, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_cities_barplots(df, title="Average Measures by City", pdf_name="Cities_Average_Measures.pdf", show=True):

    output_dir = "Barplots"
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, pdf_name)
    
    # --- sort cities by the chosen metric ---
    df_plot = df.sort_values("E[Release]", ascending=False).reset_index(drop=True)
    print(df_plot)

    with PdfPages(pdf_path) as pdf:

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        #fig.suptitle(title)

        # --- E[Release] ---
        axes[0].bar(df_plot["City"], df_plot["E[Release]"])
        #axes[0].set_title("Average Energy Released")
        axes[0].set_xlabel("City")
        axes[0].set_ylabel("Average Energy Released (Wh)")

        # --- P(NoService) ---
        axes[1].bar(df_plot["City"], df_plot["P(NoService)"])
        #axes[1].set_title("QoS Delay Probability")
        axes[1].set_xlabel("City")
        axes[1].set_ylabel("QoS Delay Probability")


        # --- E[Loss] ---
        axes[2].bar(df_plot["City"], df_plot["E[Loss]"])
        #axes[2].set_title("Average Energy Loss")
        axes[2].set_xlabel("City")
        axes[2].set_ylabel("Average Energy Loss (Wh)")

        for ax in axes:
            ax.tick_params(axis="x", rotation=30)

        # Save current figure into the PDF
        pdf.savefig(fig, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Barplots saved to: {pdf_path}")

def plot_cities_scatter(df, title="Release vs QoS (size/color = Loss)", pdf_name="Cities_Tradeoff_Scatter.pdf", show=True):

    output_dir = "Barplots"
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:

        x = df["E[Release]"].to_numpy(dtype=float)
        y = df["E[Loss]"].to_numpy(dtype=float)
        noService = df["P(NoService)"].to_numpy(dtype=float)
        #loss = df["E[Loss]"].to_numpy(dtype=float)

        # Map loss -> point size
        noService_min, noService_max = noService.min(), noService.max()
        if noService_max > noService_min:
            sizes = 80 + 420 * (noService - noService_min) / (noService_max - noService_min)
        else:
            sizes = np.full_like(noService, 200.0)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))

        sc = ax.scatter(
            x, y,
            s=sizes,
            c=noService
        )

        #ax.set_title(title)
        ax.set_xlabel("Average Energy Released (Wh)")
        ax.set_ylabel("Average Energy Loss (Wh)")

        # Annotate each city
        for i, city in enumerate(df["City"]):
            ax.annotate(
                city,
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(6, 4)
            )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("QoS Delay Probability")

        fig.tight_layout()

        # Save figure to PDF
        pdf.savefig(fig, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Scatter plot saved to: {pdf_path}")

def analyze_PV_Model():
    
    ANALYZE = 1   # 1 => Heatmaps
                  # 2 => Comparaision cities
    BufferSize = 45

    if ANALYZE == 1 : 
        """------------ Analyzing optimal policy for a single City -----------------"""
        
        """---------1) XBorne SISDMDP Generation -----------------"""
        #cityNames  = ["Barcelona", "Paris", "Rabat", "Unalaska", "Moscow"] 
        """
        Paramètres Optimal Policy Heatmamps: Buffer 45, Packet_size = 800Wh, 56 actions, 184 superstates
        r1_by_M = [0.2, 0.4, 0.6, 0.8] #[0.2, 0.4, 0.6, 0.8] # VeryCloudy, Cloudy, Partialy, Clear
        r2 = -0.5    #-0.5   #EP loss (OverFull)
        r3 = -1e3    #-1e3   #DP no-service
        """
        city    = "Barcelona"
        r1_by_M = [0.2, 0.4, 0.6, 0.8] # VeryCloudy, Cloudy, Partialy, Clear
        r2      = -0.5     #-0.5   #EP loss (OverFull)
        r3      = -1e3     #-1e3   #DP no-service
        ALL_P, states, R, N, A, Parts, weather_hours_packets, h_deb, h_fin, n_packets,\
        packet_size, pService_hourly, pSellAction = generate_XBorne_SISDMDP(city, r1_by_M, r2, r3, BufferSize)

        """----------- 2) Solving Average reward criteria ----------"""
        relative_state = Parts[0][0]  #Only for Average reward !
        rho_RB, policy_RRB, time_RRB, k_RRB  = average_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N, A, Parts, relative_state)
        energy, loss, noService = average_Measures(ALL_P, states, Parts, policy_RRB, N,
                        packet_size, BufferSize, n_packets,
                        h_deb, h_fin, pSellAction,
                        pService_hourly, weather_hours_packets)
        print("r1_by_M =",r1_by_M," r2=",r2, " r3=",r3)
        print(f'--> RPI_RB  : rau   = {rho_RB} , time = {time_RRB}, iter = {k_RRB}')
        print(f'--> Mean EP Release = {energy}')
        print(f'--> Mean EP Loss    = {loss}')
        print(f'--> QoS-Delay Prob  = {noService}')

        actions_uniques = list(dict.fromkeys(policy_RRB))
        print(actions_uniques)

        """ ---------- 3) Plot HeatMaps ----------------------------"""
        regimes_names = {0:"Very Cloudy", 1:"Cloudy", 2:"Partly cloudy", 3:"CLear sky"} 


        """
        plot_policy_heatmaps(
            states=states,
            policy_idx=policy_RRB,
            h_deb=h_deb,
            h_fin=h_fin,
            BufferSize=BufferSize,
            city=city,
            r1_by_M=r1_by_M,
            r2=r2,
            r3=r3,
            regimes_names=regimes_names,
            show=False   # recommandé
        )
        """

        plot_policy_decision_maps_by_M_pdf_2x2(
            states=states,
            policy_idx=policy_RRB,
            h_deb=h_deb,
            h_fin=h_fin,
            city = city,
            r2 = r2,
            r3 = r3,
            regimes_names=regimes_names,
            D_max=BufferSize,
            BufferSize=BufferSize,
            show=False,
            dpi=150
        )

        plot_policy_threshold_summary_by_M_pdf(
            states=states,
            policy_idx=policy_RRB,
            h_fin=h_fin,
            city=city,
            r2=r2,
            r3=r3,
            pSellAction=pSellAction,
            BufferSize=BufferSize,
            D_max=BufferSize,
            regimes_names=regimes_names,
            avg_release=energy,
            avg_loss=loss,
            qos_delay_prob=noService,
            show=False,
            dpi=150
        )

    if ANALYZE == 2:
        """------------ Comparing optimal measures for several cities -----------------"""

        cityNames  = ["Barcelona", "Paris", "Rabat", "Unalaska", "Moscow", "Tokyo","London","Reykjavik", "Athens", "Yaounde", "Santiago"]

        # Same reward parameters for all cities
        r1_by_M = [0.2, 0.4, 0.6, 0.8]
        r2      = 0 #-0.5
        r3      = 0 #-1e3

        results = []

        for city in cityNames:
            print(f"\n===== Processing city: {city} =====")

            # 1) Generate model
            ALL_P, states, R, N, A, Parts, weather_hours_packets, h_deb, h_fin, n_packets,\
            packet_size, pService_hourly, pSellAction = generate_XBorne_SISDMDP(
                city, r1_by_M, r2, r3, BufferSize
            )

            # 2) Solve average-reward MDP
            relative_state = Parts[0][0]
            rho_RB, policy_RRB, time_RRB, k_RRB = average_policy_Iteration_Csr_Chiu_Rob(
                ALL_P, R, N, A, Parts, relative_state
            )

            # 3) Compute average measures
            energy, loss, noService = average_Measures(
                ALL_P, states, Parts, policy_RRB, N,
                packet_size, BufferSize, n_packets,
                h_deb, h_fin, pSellAction,
                pService_hourly, weather_hours_packets
            )

            print(f"rho = {rho_RB:.4f}, time = {time_RRB:.2f}s, iters = {k_RRB}")
            print(f"E[Release] = {energy:.4f}")
            print(f"E[Loss]    = {loss:.4f}")
            print(f"P(NoService) = {noService:.6f}")

            results.append({
                "City": city,
                "rho": rho_RB,
                "E[Release]": energy,
                "E[Loss]": loss,
                "P(NoService)": noService,
            })

        # ---------- Convert to DataFrame ----------
        df = pd.DataFrame(results)
        plot_cities_barplots(df,
            title="PV Model — Average Measures by City",
            pdf_name=f"B{BufferSize}_Cities_Average_Measures_r2_{r2}_r3_{r3}.pdf",
            show=False
        )

        plot_cities_scatter(df,
            title="Trade-off: Energy Sold vs Energy Loss vs Delay (size/color)",
            pdf_name=f"B{BufferSize}_Cities_Tradeoff_r2_{r2}_r3_{r3}.pdf",
            show=False
        )


'''A) To analyse the Xborne SISDMDP based on real data from NREL'''
analyze_PV_Model()

'''B) To test results of each algorithm : one experiment for each (N, K, A). Synthetic Data'''
#test_scalability_algorithms() 

'''C) To test results of each algorithm : 30 experiments for each (N, K, A). Synthetic Data'''
#test_scalability_algorithms_CI95(30)
