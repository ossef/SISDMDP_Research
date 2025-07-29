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

import time 
import numpy as np
from scipy.sparse import csr_matrix
from scipy import stats

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

def generate_Synthetic_SISDMDP(k, N, A, epsilon=1e-3):
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

def test_algorithms() :
    """------------1) SISDMDP Generation -----------------------"""
    K, N, A = 10, 5000, 200
    ALL_P, R, N, A, Parts = generate_Synthetic_SISDMDP(K, N, A)

    """----------- 2) Solving Average reward criteria ----------"""
    relative_state = Parts[0][0]  #Only for Average reward !
    rho_RB, policy_RRB, time_RRB, k_RRB  = average_policy_Iteration_Csr_Chiu_Rob(ALL_P, R, N, A, Parts, relative_state)
    rho_RC, policy_RC, time_RC, k_RC     = average_policy_Iteration_Csr_Chiu(ALL_P, R, N, A, Parts, relative_state)
    rho_GJ, policy_RGJ, time_RGJ, k_RGJ  = average_policy_Iteration_Csr_GJ(ALL_P, R, N, A, relative_state)
    rho_FP, policy_RFP, time_RFP, k_RFP  = average_policy_Iteration_Csr_FP(ALL_P, R, N, A, relative_state)
    rho_RVI, policy_RVI, time_RVI, k_RVI = average_relative_Value_Iteration_Csr(ALL_P, R, N, A, relative_state)

    print(f'\nAVG Results : K={K}, N={N}, A={A}')
    print(f'--> RPI_RB  : rau = {rho_RB} , time = {time_RRB}, iter = {k_RRB}')
    print(f'--> RPI_RC  : rau = {rho_RC} , time = {time_RC}, iter = {k_RC}')
    print(f'--> RPI_GJ  : rau = {rho_GJ} , time = {time_RGJ}, iter = {k_RGJ}')
    print(f'--> RPI_FP  : rau = {rho_FP} , time = {time_RFP}, iter = {k_RFP}')
    print(f'--> RVI     : rau = {rho_RVI} , time = {time_RVI}, iter = {k_RVI}')

    vals = np.array([rho_FP, rho_GJ, rho_RB, rho_RVI, rho_RC])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e  for a,b,c,d,e in zip(policy_RFP,policy_RGJ,policy_RRB,policy_RVI, policy_RC)) ):
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

def test_confidence_interval_normalized(n_runs):
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
        ALL_P, R, N_gen, A_gen, Parts = generate_Synthetic_SISDMDP(K, N, A)
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

''' To test results of each algorithm : one experiment for each (N, K, A)'''
test_algorithms()

''' To test results of each algorithm : 30 experiments for each (N, K, A)'''
#test_confidence_interval_normalized(30)
