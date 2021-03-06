# Optimal Non-Convex Exact Recovery in Stochastic Block Model via Projected Power Method

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Optimal Non-Convex Exact Recovery in Stochastic Block Model via Projected Power Method" (accepted by ICML 2021)
by Peng Wang, Huikang Liu, Zirui Zhou, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

===========================================================================

This package contains 3 experimental tests to output the results in the paper:

* In the folder named phase-transition, we conduct the experiment of phase transition to test recovery performance and running time of our proposed PPM and compare it with SDP-based method in Amini et al. (2018), the spectral clustering (SC) method in Su et al. (2019), and the local penalized ML estimation (PMLE) method in Gao et al. (2017).
  - demo_phase_transition.m: Output the recovery performance and running time of above methods
  - PPM.m: Implement our method in Algorithm 1
  - PMLE.m: Implement the local penalized ML estimation method in Gao et al. (2017)
  - sdp_admm1.m: Implement SDP-based approach by alternating direction method of multipliers (ADMM) in Amini et al. (2018)
  - SC.m: Implement the spectral clustering method in Su et al. (2019) via MATLAB function eigs
  - LP_gurobi.m: Solve the projection problem in projected power iterations via GUROBI linear programming solver 
  - LP_MCAP.m: Solve the projection problem in projected power iterations via the method in Tokuyama & Nakano (1995) 
  - spectral_init.m: Compute the initial point via Algorithm 2 in Gao et al. (2017)

* In the folder named convergence-performance, we conduct the experiments of convergence performance to test the number of iterations needed by our proposed
PPM to exactly identify the underlying communities.
  - demo_convergence.m: Output the convergence performance of our method PPM
  - dist_to_GD.m: Compute the distance from an iterate to a ground truth

* In the folder named recovery-efficiency-accuracy, we compare the recovery efficiency and accuracy of our proposed method with SDP, SC, and PMLE on real data sets. 
  - test_synthetic.m: Output the results on synthetic data sets
  - test_polbooks.m: Output the results on real data set polbooks
  - test_polblogs.m: Output the results on real data set polblogs
  - misclassify_points.m: Compute the number of misclassified vertices of an iterate compared to a ground truth 
  - The folder named Datasets contains three real data sets, plobooks, polblogs and football, which is downloaded from the SuiteSparse Matrix Collection (Davis & Hu, 2011).
