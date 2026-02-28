# SISDMDP framework

## I - Goal

This framework provides an efficient methodology for computing optimal policies in structure-based Markov Decision Processes (MDPs), focusing on a class of problems we define as SISDMDPs (Single-Input Superstate Decomposable MDPs), introduced in [1].

It supports both average-reward and discounted-reward criteria, as well as CSR sparse representations. The SISDMDP framework generalizes the structured models analyzed in previous work (<a href="https://github.com/ossef/MDP_Battery" target="_blank">MDP Battery</a>), providing a scalable solution for large-scale decomposable decision-making problems.

The proposed solution method is applied to a data-driven photovoltaic energy storage application based on NREL data [2]. The objective is to jointly optimize multiple performance metrics, including energy release, QoS delay, and energy loss. The framework enables a detailed structural analysis of the optimal policy and compares average performance metrics across cities with heterogeneous meteorological conditions.

## II - Project architecture

Tree of the most important files and folders in the project :

```
/
├─┬─Models/: Stores all Battery SISDMDP models
│ ├─── Toy_example_Model/   # Small illustrative SISDMDP used for visualizing the transition kernel structure.
│ └─┬─ SIMPA_Journal_Model/ # Realistic NREL-based data-driven SISDMDP models generated using XBorne.
│   ├─-NREL_Data            # Raw photovoltaic production data for multiple cities (Rabat, Paris, Barcelona, Moscow, Unalaska ...).
│   ├─-NREL_Extracts        # Extracted discretized distributions derived from 'NREL_Data' for each city (run 'Dist_gener.py' file).
│   ├─-NREL_Models          # The XBorne generated SISDMDP models that uses 'NREL_Extracts' distributions (run 'scriptMDP' file).
│   └─-Dist_gener.py        # Preprocessing script converting raw PV data in 'NREL_Data' into discretized distributions in 'NREL_Extracts'.
│
├─┬─Solve_SISDMDP/: Core algorithms for SISDMDP solving.
│ ├── generate_matrix.py        # Generates synthetic SISDMC-SC structured Markov chains.
│ ├── gth_full.py               # GTH algorithm for steady-state probability distribution computation.
│ ├── chiu_classic.py           # Feinberg & Chiu algorithm for steady-state probability distribution.
│ ├── chiu_ROB.py               # Proposed algorithm (Chiu + Rob-B) for steady-state probability distribution.
│ ├─- Graph.py                  # Reads ".Rii" matrix storage from '/Models' folder, convert to sparse_row
│ ├── Solve_AVG_DSC_Reward.py   # Implementation of MDPs resolution :
│ │                                 # 5 average reward algorithms.
│ │                                 # 4 discounted reward algorithms.
│ │                                 # The proposed approches are: MRPI + Chiu, MRPI + Chiu+ RobB, and PI + Chiu+ RobB.
│ └──Launcher.py                # Main experimentation script :
│                                 A) Scalability testing on synthetic SISDMDPs.
│                                 B) Loading real NREL-based SISDMDPs (from '../Models/SIMPA_Journal_Model/NREL_Models/').
│                                 C) Solving and visualizing results  (stored "../Results/" folder).
├─── Results/: 
│       ├─- HeatMaps    # Detailed optimal policy visualisation. 
│       └── Barplots    # Average optimal performance measures and trade-off visualizations.
│
├───ScreenShots/        # Screenshots for below explanations
└───README.md           # Framework description and instructions
```

## III - Usage 
- All Markov Decision Process (MDP) algorithms are implemented in `Solve_AVG_DSC_Reward.py`
- Markov chain algorithms are implemented in `gth_full.py`, `chiu_classic.py`, and `chiu_ROB.py`
- The main launching file is `Launcher.py`, it contains:
    - test_algorithms():
        - Generates a synthetic SISDMDP model (using function from `generate_matrix.py`), with parameters:
            - N: number of states
            - K: number of superstates
            - |A|: number of actions
        - Executes:
            - Up to 5 Average-reward algorithms
            - Up to 4 Discounted-reward algorithms
        - Outputs:
            - Execution time
            - Number of iterations
            - Average reward (for average criteria)
            - Comparison of optimal policies
        - You can comment/uncomment specific algorithms in the code to customize the tests
    - test_confidence_interval_normalized(n_runs):
        - Repeats the test_algorithms() experiments n_runs times to estimate 95% confidence intervals.
    - analyze_PV_Model():
      - Reads the data-driven SISDMDP model (stored in `/Models/SIMPA_Journal_Model/NREL_Model/`)
      - Solves the model and extracts the optimal policy
      - Computes and plots average optimal measures, either focusing and details on a single city (ANALYZE=1) or comparing results for different cities (ANALYZE=2).
      

## IV - Scalability testing of algorithms: 

Results of ($|A|=200$, $N=5000$, $K=10$) configuration: 
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

## V - Data-driven PV energy storage application: 
The formal description of the data-driven SISDMDP model is described in [2]. To run the application we will have to preprocess PhotoVoltaic energy production of Raw data (`/Models/SIMPA_Journal_Model/NREL_Data`). 

### Data-driven SISDMDP generation and solving

1) Execute the python code 'Dist_gener.py' in `/Models/SIMPA_Journal_Model/`: this code tranforms the Raw data of each city to discrete units energy distribution for each hour of the day, clustered by four wheater regimes (Very cloudy, Cloudy, Partly cloudy, Clear sky). After execution one can find the distributions in `/Models/SIMPA_Journal_Model/NREL_Extracts`. To test different scenarios, one can modify parameters in header part as Packets size and active hours interval. The average execution time (M1 laptop) is ~ 80 seconds, that inlcudes distributions generations for 11 cities.

2) Execute 'scriptMDP' file in `/Models/SIMPA_Journal_Model/NREL_Model`: this script generates actions (generActions.py) then uses the C code of <a href="https://github.com/ossef/XBorne-2017" target="_blank">XBorne</a> framework to generate the data-driven SISDMDP based on distributions in `/Models/SIMPA_Journal_Model/NREL_Extracts`. After execution, the ready to solve models are therefore stored in `/Models/SIMPA_Journal_Model/NREL_Model` in cities named folders. Due to GitHub size limitations, only Athens and Barcelona models are included (full project size 20 GB). To test different scenarios, one can modify the Buffersize or actions generations file. However, to alter the kernel structure of the MDP, manipulation of XBorne is required, particularly with 'fun.c' and 'const.h' file, which encodes the structure of a Markov Chain, including the description of states, various events, transitions, and their probabilities. The average execution time of this script is ~ 5 minutes for 11 cities where each model contains 56 actions, each action describe ~ 45,000 states, ~ 250,000 transitions, and 184 superstates.

3) Verify that `/Models/SIMPA_Journal_Model/NREL_Model` includes models. One can run two types of experiments in 'analyze_PV_Model()' function of  `/Solve_SISDMDP/Launcher.py`. First, define the rewards variables r1_by_M (Energy Packet release), r2 (EP-Energy-Packet loss penalty) and r3 (DP-Data-Packet delay) then set the experiment mode:
   - ANALYZE = 1: fixe the name of a SISDMDP city model and run `Launcher.py` file. The detailled results of the optimal policy will be stored in `/Results/HeatMaps/`.
   - ANALYZE = 2: fixe the name of cities to compare, then run `Launcher.py` file. The detailled results of average measures comparisons will be stored in `/Results/Barplots/`.

### Data-driven SISDMDP execution and results
Run the following scenarios explained in Section 7.2 of [2].
 - Agressive DP delay minimization       : r1_by_M= [0,0,0,0],         r2= 0,    r3= -1000
 - Agressive EP overflow minimization    : r1_by_M= [0,0,0,0],         r2= -0.5, r3= 0
 - Regime dependent EP selling           : r1_by_M= [0.2,0.4,0.6,0.8], r2= 0,    r3= 0
 - Balanced Multi-Objective Optimization : r1_by_M= [0.2,0.4,0.6,0.8], r2= -0.5, r3= -1000

#### Results of "ANALYZE=1" experiment mode with Barcelona
Detailled analysis of the optimal policy of the average reward SISDMDP agent
<div align="center">
    <img src="ScreenShots/Optimal_policy.png" width="800" height="650"/>
</div>
<br>

#### Results of "ANALYZE=2" experiment mode, on last two scenarios for different cities

Cross-city average performance comparison and trade-off analysis of released energy, overflow losses, and QoS delay probability
<div align="center">
    <img src="ScreenShots/Cities_Scatter_heatmap.png" width="900" height="650"/>
</div>
<br>
along with detailled barplots for "Regime dependent EP selling" scenario
<br>
<div align="center">
    <img src="ScreenShots/Barplot_Scen3.png" width="920" height="670"/>
</div>
and "Balanced Multi-Objective Optimization" scenario
<div align="center">
    <img src="ScreenShots/Barplot_Scen4.png" width="920" height="670"/>
</div>

##  Contributors
- [Youssef AIT EL MAHJOUB](https://github.com/ossef)
- [Salma Alouah](https://github.com/salouah003)

Preprint [1]: "Efficient Solving of Large Single Input Superstate Decomposable Markovian Decision Process". Youssef AIT EL MAHJOUB, Jean-Michel FOURNEAU and Salma ALOUAH. https://arxiv.org/abs/2508.00816. 2025.

Submitted [2]: "Efficient Solving of Large Single Input Superstate Decomposable Markovian Decision Process with Application to Photovoltaic Energy Storage". Youssef AIT EL MAHJOUB, Jean-Michel FOURNEAU and Salma ALOUAH. 2026.

Some related works: <br> 
<a href="https://doi.org/10.1016/j.comcom.2025.108273" target="_blank">ComCom2025 journal: data-driven PV model and structured MDPs</a> <br>
<a href="https://ieeexplore.ieee.org/abstract/document/10770514/" target="_blank">WiMob2024 conference: On/Off PV model and structured MDPs</a> <br>
<a href="https://www.researchgate.net/publication/331334323_A_numerical_approach_of_the_analysis_of_optical_container_filling" target="_blank">ValueTools2019 conference: NGreen optical container and structured Markov chain</a> <br>
<a href="https://www.researchgate.net/publication/329954281_Performance_and_energy_efficiency_analysis_in_NGREEN_optical_network " target="_blank">WiMob2018 conference: NGreen optical network</a> 











