
%% set the path of gurobi for solving LP of PPM
addpath('D:\Software\gurobi903\win64\matlab');
clear; clc;

%% Load the Real-World Dataset
load Datasets/polblogs          %%% load the set polblogs
As = Problem.A; As(As > 0) = 1; %%% As: adjacency matrix
xt = Problem.aux.nodevalue;     %%%%% xt: ground truth
n = size(As,1); 

%% determine the number of communities and each community size
K = 2; pi = size(K,1);
pi(1) = sum(xt==0); pi(2) = sum(xt==1); 
xt = (xt - 0.5)*2; 

%% form the clustering matrix of the ground truth
Ht = zeros(n,K); Ht(1:pi(1),1) = 1; Ht(pi(1)+1:n,2) = 1;

%% parameters setting
iternum = 10;
[fval_PPM, fval_MLE, fval_SDP] = deal(zeros(iternum,1));
[PPM_collector, MLE_collector] =  deal(zeros(n,K,iternum));
[ttime_PPM, ttime_MLE, ttime_SDP, ttime_SC] = deal(0);
SDP_collector = zeros(n,n,iternum);

%% choose the running algorithms
run_PPM = 1; run_MLE = 1; run_SDP = 1; run_SC = 1;

for iter = 1:iternum
    
        fprintf('Iter Num: %d \n', iter);
        
        %% generate the initial point
        H0 = randn(n,K);
        maxiter = 1e3; tol = 1e-3; report_interval = 1e1; total_time = 1e3;
                
        %% Spectral Clustering (SC)
        if run_SC == 1
            tic; e_SC = SC((As+As')/2, n, K); time_SC = toc;
            ttime_SC = ttime_SC + time_SC; 
        end

        %% Projected Gradient Method (PPM)
        if run_PPM == 1
            H0 = LP_gurobi(H0, n, pi', K);
            opts = struct('T', maxiter, 'tol', tol, 'report_interval', 1, 'quiet', 1);
            tic; H_PPM = PPM(As, H0, pi, opts); time_PPM=toc;
            ttime_PPM = ttime_PPM + time_PPM; H_PPM = sparse(H_PPM);
            PPM_collector(:,:,iter) = H_PPM;
            fval_PPM(iter) = -trace(H_PPM'*As*H_PPM);
        end
        
        %% Solve the SDP by ADMM             
        if run_SDP == 1
            X0 = H0*H0'; 
            opts = struct('rho', 3e0, 'T', maxiter, 'tol', tol, 'report_interval', report_interval, 'quiet', 1);
            tic; X_SDP = sdp_admm1(As, X0, K, opts); time_SDP=toc;
            ttime_SDP = ttime_SDP + time_SDP;
            SDP_collector(:,:,iter) = X_SDP;
            fval_SDP(iter) = -trace(sparse(X_SDP)'*As);
        end        
        
        %% Penalized MLE based method (PMLE)
        if run_MLE == 1
            tic; H_MLE = PMLE(As, n, K); time_MLE=toc;
            ttime_MLE = ttime_MLE + time_MLE; H_MLE = sparse(H_MLE);
            MLE_collector(:,:,iter) = H_MLE;
            fval_MLE(iter) = -trace(H_MLE'*As*H_MLE);
        end

end 

%% Penalized MLE based method
color_choice = 8;
subplot(2,3,1); imagesc(As); title('Adjacency matrix');
subplot(2,3,2); imagesc(xt*xt'); title('Ground truth of Polblogs');

%% compute the misclassified vertices of each method
if run_PPM == 1
    [min_PPM, inx] = min(fval_PPM); H_PPM = PPM_collector(:,:,inx);
    X_PPM = H_PPM*H_PPM'; e_PPM = labelsFromX(X_PPM,K);     
    e_PPM = (e_PPM - 1.5)*2;
%     dist_PPM = min(nnz(e_PPM-xt), nnz(e_PPM+xt));
    mis_points_PPM = misclassify_points(H_PPM,Ht);
    subplot(2,3,3); imagesc(e_PPM*e_PPM'); title('PPM'); 
    fprintf('MVs of PPM: %d, time of PPM: %.2f \n', mis_points_PPM, ttime_PPM);
end

if run_MLE == 1
    [min_MLE, inx] = min(fval_MLE); H_MLE = MLE_collector(:,:,inx);
    X_MLE = H_MLE*H_MLE'; e_MLE = labelsFromX(X_MLE,K); 
    H_MLE = zeros(n,K); 
    for k = 1:K
       inx = find(e_MLE == k); H_MLE(inx,k) = 1;
    end   
    mis_points_MLE = misclassify_points(H_MLE,Ht);
    e_MLE = (e_MLE - 1.5)*2;
%     dist_MLE = min(nnz(e_MLE-xt), nnz(e_MLE+xt));    
    subplot(2,3,4); imagesc(e_MLE*e_MLE'); title('PMLE'); 
    fprintf('MVs of PMLE: %d, time of PMLE: %.2f \n', mis_points_MLE, ttime_MLE);
end

if run_SDP == 1
    [min_SDP, inx] = min(fval_SDP); X_SDP = SDP_collector(:,:,inx);
    e_SDP = labelsFromX(X_SDP,K); 
    H_SDP = zeros(n,K); 
    for k = 1:K
       inx = find(e_SDP == k); H_SDP(inx,k) = 1;      
    end
    time_SDP = ttime_SDP/iternum; e_SDP = (e_SDP - 1.5)*2;
%     dist_SDP = min(nnz(e_SDP-xt), nnz(e_SDP+xt));   
    mis_points_SDP = misclassify_points(H_SDP,Ht);
    subplot(2,3,5); imagesc(e_SDP*e_SDP'); title('SDP'); 
    fprintf('MVs of SDP: %d, time of SDP: %.2f \n', mis_points_SDP, ttime_SDP);
end

if run_SC == 1
    time_SC = ttime_SC/iternum; 
    H_SC = zeros(n,K);
    for k = 1:K
       inx = find(e_SC == k); H_SC(inx,k) = 1;       
    end
    mis_points_SC = misclassify_points(H_SC,Ht);
    e_SC = (e_SC - 1.5)*2;
%     dist_SC = min(nnz(e_SC-xt), nnz(e_SC+xt));    
    subplot(2,3,6); imagesc(e_SC*e_SC'); title('SC'); 
    fprintf('MVs of SC: %d, time of SC: %.2f \n', mis_points_SC, ttime_SC);
end



    