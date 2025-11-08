# SISDMDP framework

## I - Goal

This framework provides an efficient method for computing the optimal policy in structure-based Markov Decision Processes (MDPs), focusing on a class of problems we define as SISDMDPs (Single Input State Decomposable MDPs), introduced in [1].
It supports both `average` and `discounted` reward criteria, and csr_sparse models. The SISDMDP framework generalizes the structured models analyzed in previous work <a href="https://github.com/ossef/MDP_Battery" target="_blank"> MDP_Battery </a>, offering a scalable solution suitable for large-scale decomposable decision-making problems.


## II - Project architecture

Tree of the most important files and folder in the project's repository :

```
/
├───ScreenShots/: contains some screenshots for below explanations
├─┬─Solve/: contains the source code for the MDP solving
│ ├── generate_matrix.py        # Generating a SISDMC-SC structured Markov chain to use in "Launcher.py"
│ ├── gth_full.py               # GTH algorithm: steady-state probability distribution
│ ├── chiu_classic.py           # Chiu and Feinberg algorithm: for steady-state probability distribution
│ ├── chiu_ROB.py               # Proposed algorithm (Chiu + Rob-B): for steady-state probability distribution
│ ├── Solve_AVG_DSC_Reward.py   # Solving MDPs: average reward (5 algorithms) and discounted reward (4 algorithms)
│ │                             # The proposed algorithms are: MRPI + Chiu, MRPI + Chiu+ RobB, and PI + Chiu+ RobB
│ └──Launcher.py                # Script to run the experiments on synthetic generated SISDMDPs 
│
└───README.md                   # Framework description and instructions
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

## IV - Results of ($|A|=200$, $N=5000$, $K=10$) configuration: 

- In Discounted $(\gamma=0.9)$ reward, with `test_algorithms()` function: 
<br>
<div align="center">
    <img src="ScreenShots/ScreenShot_DSC.png" width="500" height="140"/>
</div>
<br>

- In Average reward, with `test_algorithms()` function: 
<br>
<div align="center">
    <img src="ScreenShots/ScreenShot_AVG.png" width="500" height="90"/>
</div>
<br>

- In Discounted $(\gamma=0.9)$ reward, with `test_confidence_interval_normalized(n_runs=30)` function: <br>
(Each algorithm is run 30 times)
<br>
<div align="center">
    <img src="ScreenShots/Results_IC_95.png" width="500" height="75"/>
</div>
<br>

##  Contributors

- [Youssef AIT EL MAHJOUB](https://github.com/ossef)
- [Salma Alouah](https://github.com/salouah003)

Original article [1]: <br>"Efficient Solving of Large Single Input Superstate Decomposable Markovian Decision Process", Youssef AIT EL MAHJOUB, Jean-Michel FOURNEAU and Salma ALOUAH. Pre-print document, https://arxiv.org/abs/2508.00816, 2025.

Some related works: <br> 
https://doi.org/10.1016/j.comcom.2025.108273<br>
https://ieeexplore.ieee.org/abstract/document/10770514/ <br>
https://www.researchgate.net/publication/331334323_A_numerical_approach_of_the_analysis_of_optical_container_filling <br>
https://www.researchgate.net/publication/329954281_Performance_and_energy_efficiency_analysis_in_NGREEN_optical_network 


