# Optimal Non-Convex Exact Recovery in Stochastic Block Model via Projected Power Method

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Optimal Non-Convex Exact Recovery in Stochastic Block Model via Projected Power Method" (accepted by ICML 2021)
by Peng Wang, Huikang Liu, Zirui Zhou, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

===========================================================================

This package contains 3 experimental tests to output the results in the paper:

* In the folder named phase-transition, we conduct the expriment of phase transition to test recovery performance of our approach PPM and compare it with SDP-based approach in Amini et al. (2018), the manifold optimization (MFO) based approach in Bandeira et al. (2016), and the spectral clustering (SC) approach in Abbe et al. (2017).
  - demo_phase_transition.m: Output the recovery performance and running time of above methods
  - PPM.m: Implement our method in Algorithm 1
  - PMLE: Implement the local penalized ML estimation method in Gao et al. (2017)
  - sdp_admm1: Implement SDP-based approach by alternating direction method of multipliers (ADMM) in Amini et al. (2018)
  - SC: Implement the spectral clustering method in Su et al. (2019) via MATLAB function eigs

* In the folder named convergence-performance, we conduct the experiments of convergence performance to test the number of iterations needed by our approach
PPM to exactly identify the underlying communities.
  - demo_convergence.m: Output the convergence performance of our method PPM
  - dist_to_GD.m: Compute the distance from an iterate to a ground truth, i.e., $$min_{Q \in \Pi_K} \|H-H^* Q\|_F$$.

* In the folder named computational-efficiency, we compare the computational efficiency of our proposed method with GPM, MGD, SDP, and SC on both synthetic and real data sets. 
  - GPM.m: Implement the two-stage method consisting of power iterations (PIs) + generalized power iterations (GPIs) in Wang et al. (2020). 
  - test_synthetic.m: Output the results on synthetic data sets
  - test_polbooks.m: Output the results on real data set polbooks
  - test_polblogs.m: Output the results on real data set polblogs

![](http://latex.codecogs.com/gif.latex?\\frac{\\partial J}{\\partial \\theta_k^{(j)}}=\\sum_{i:r(i,j)=1}{\\big((\\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\\big)x_k^{(i)}}+\\lambda \\xtheta_k^{(j)})

