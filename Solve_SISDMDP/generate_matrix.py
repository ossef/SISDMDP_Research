from scipy.sparse import csr_matrix
import numpy as np
import random
import time

def generate_sparse_matrix(k, N, epsilon=1e-3):
    """ Generates a synthetic SISDMC-SC structured Markov chain """
    L = N//k
    start = time.time()

    numbers = list(range(N))
    superstates = [numbers[i * L:(i + 1) * L] for i in range(k)]
    #print(superstates)
    single_inputs = [s[0] for s in superstates]

    data, rows, cols = [], [], []

    for superstate_idx in range(k):
        current_superstate = superstates[superstate_idx]
        current_single_input = current_superstate[0]
        reached_states = np.zeros(L, dtype=bool)

        for state_idx in range(L):
            current_state = current_superstate[state_idx]
            P_interne = epsilon + (1 - epsilon) * random.random()

            data_tmp, cols_tmp = [], []

            # --- Transitions internes ---
            all_internal = current_superstate[state_idx + 1:] if state_idx < L - 1 else []
            m = len(all_internal)
            if state_idx == 0:
                limite = random.randint(1 , m)
                possible_states = all_internal[:limite] 

            else:
                if m > 0:
                    limite = random.randint(1 , m)
                    possible_states = all_internal[:limite] + [current_single_input]
                else:
                    possible_states = [current_single_input]

            tirages = np.random.rand(len(possible_states))
            proba = (tirages / tirages.sum()) * P_interne

            for idx, target in enumerate(possible_states[:-1]):
                target_adapted= target % L
                if not reached_states[target_adapted] or random.random() < 0.001:
                    cols_tmp.append(target)
                    data_tmp.append(proba[idx])
                    reached_states[target_adapted] = True

            # Transition vers single-input
            cols_tmp.append(current_single_input)
            data_tmp.append(proba[-1])
            reached_states[(current_single_input % L)] = True

            # --- Forçage de connexité ---
            while not np.all(reached_states):
                unreachable = [j for j in range(L) if not reached_states[j]]
                if not unreachable:
                    break
                for j in unreachable:
                    j_global = current_superstate[j]
                    i_candidates = [i for i in range(L) if reached_states[i]]
                    
                    if i_candidates:
                        cols_tmp.append(j_global)
                        data_tmp.append(random.uniform(0.01, 0.05))
                        reached_states[j] = True


            # --- Transitions externes ---
            P_externe = 1 - P_interne
            if state_idx > 0:
                ext_inputs = [s for i, s in enumerate(single_inputs) if i != superstate_idx]
                tirages_ext = np.random.rand(len(ext_inputs))
                proba_ext = (tirages_ext / tirages_ext.sum()) * P_externe

                for idx, target in enumerate(ext_inputs):
                    if random.random() < 0.05:
                        cols_tmp.append(target)
                        data_tmp.append(proba_ext[idx])
            else:
                next_input = superstates[(superstate_idx + 1) % k][0]
                cols_tmp.append(next_input)
                data_tmp.append(P_externe)

            # --- Normalisation directe ---
            total = sum(data_tmp)
            data_tmp = [val / total for val in data_tmp]

            for col, val in zip(cols_tmp, data_tmp):
                rows.append(current_state)
                cols.append(col)
                data.append(val)

    # Construction finale de la matrice sparse
    P_csr = csr_matrix((data, (rows, cols)), shape=(N, N))

    end = time.time()
    print(f"\n✅ Original MC Generation time :  {end - start:.4f} sec")
    print(f"➡️ Parameters of generated matix : N={N}, K={k}, M={P_csr.nnz}")
    
    return P_csr, superstates

