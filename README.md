# SISDMDP framework

## I - Goal

This framework provides an efficient method for computing the optimal policy in structure-based Markov Decision Processes (MDPs), focusing on a class of problems we define as SISDMDPs (Single Input State Decomposable MDPs), introduced in [1].
It supports both `average` and `discounted` reward criteria, and csr_sparse models. The SISDMDP framework generalizes the structured models analyzed in previous work <a href="https://github.com/ossef/MDP_Battery" target="_blank"> MDP_Battery </a>, offering a scalable solution suitable for large-scale decomposable decision-making problems.


## II - Project architecture

Tree of the most important files and folder in the project's repository :

```
/
├── generate_matrix.py        # Generating a SISDMC-SC structured Markov chain to use in "Launcher.py"
├── gth_full.py               # GTH algorithm : steady-state probability distribution
├── chiu_classic.py           # Chiu and Feinberg algorithm : for steady-state probability distribution
├── chiu_ROB.py               # Proposed algorithm (Chiu + Rob-B) : for steady-state probability distribution
├── Solve_AVG_DSC_Reward.py   # Solving average reward (5 algorithms) and discounted reward (4 algorithms)
│                             # The proposed algorithms are : MRPI + Chiu, MRPI + Chiu+ RobB and PI + Chiu+ RobB
├── Launcher.py               # Script to run experiments on synthetic generated SISDMDPs 
└───README.md                 # Project description and instructions
```

## III - Usage 
- All Markov Decision Process (MDP) algorithms are implemented in `Solve_AVG_DSC_Reward.py`
- Markov chain algorithms are implemented in `gth_full.py`, `chiu_classic.py`, and `chiu_ROB.py`
- `Launcher.py` file contains functions to generate synthetic SISDMDP instances and perform evaluation. It includes two main testing functions:
    - test_algorithms():
        - Generates a synthetic SISDMDP model using the helper function from `generate_matrix.py`, based on the following parameters :
            - N: number of states
            - K: number of paritions
            - |A|: number of actions
        - Executes several algorithms:
            - Up to 5 Average Reward algorithms
            - Up to 4 Discounted Reward algorithms
        - Outputs:
            - Execution time
            - Number of iterations
            - Average reward (for average criteria)
            - Comparison of optimal policies
        - You can comment/uncomment specific algorithms in the code to customize the test
    - test_confidence_interval_normalized(n_runs):
        - Repeats the test_algorithms() process n_runs times to obtain a 95% confidence interval.
     
##  Contributors

- [Youssef AIT EL MAHJOUB](https://github.com/ossef)
- Salma Alouah

Original article, [1]: "Efficient Solving of Large Single Input Superstate Decomposable Markovian Decision Process", Youssef AIT EL MAHJOUB, Jean-Michel FOURNEAU and Salma ALOUAH. Submitted to, 18th EAI International Conference on Performance Evaluation Methodologies and Tools, Valuetools, 2018.

Some related articles: https://authors.elsevier.com/a/1lU75VwcQvf4y, https://ieeexplore.ieee.org/abstract/document/10770514/

          
