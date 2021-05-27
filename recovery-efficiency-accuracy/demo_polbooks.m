
%% set the path of gurobi for solving LP of PPM
addpath('D:\Software\gurobi903\win64\matlab');
clear all; clc;

%% Load the Real-World Dataset
load Datasets/polbooks; %%% load the set polbooks
As = Problem.A; As(As > 0) = 1; %%% As: adjacency matrix
xt = Problem.aux.nodevalue; %%%%% xt: ground truth

%% determine the number of communities and each community size
idx1 = find(xt == 'n'); idx2 = find(xt == 'c'); idx3 = find(xt == 'l');
n1 = size(idx1,1); n2 = size(idx2,1); n3 = size(idx3, 1); % idx2 = idx2(n1-n2+1:n1);
idx = [idx1; idx2; idx3]; As = As(idx, idx); xt = xt(idx); 
n = size(As,1); K = 3;
pi = [n1;n2;n3]; 

%% form the clustering matrix of the ground truth
Ht = sparse(zeros(n,K)); 
Ht(1:n1,1) = 1; Ht(n1+1:n1+n2,2) = 1; Ht(n1+n2+1:n,3) = 1;

%% parameters setting
iternum = 10;
[fval_PPM, fval_MLE, fval_SDP] = deal(zeros(iternum,1));
[PPM_collector, MLE_collector] =  deal(zeros(n, K, iternum));
[ttime_PPM, ttime_MLE, ttime_SDP, ttime_SC] = deal(0);
SDP_collector = zeros(n,n,iternum);

%% choose the running algorithms
run_PPM = 1; run_MLE = 1; run_SDP = 1; run_SC = 1;

for iter = 1:iternum
    
        fprintf('Iter Num: %d \n', iter);
        %% generate the initial point
        rng(iter+10); H0 = randn(n,K);
        maxiter = 1e3; tol = 1e-3; total_time = 1e3;
        
        %% Penalized MLE based method (PMLE)
        if run_MLE == 1
            tic; H_MLE = PMLE_1(As, n, K); time_MLE=toc;
            ttime_MLE = ttime_MLE + time_MLE; H_MLE = sparse(H_MLE);
            X_MLE = H_MLE*H_MLE'; MLE_collector(:,:,iter) = H_MLE;
            fval_MLE(iter) = -trace(H_MLE'*As*H_MLE);
        end
                
        %% Spectral Clustering (SC)
        if run_SC == 1
            tic; e_SC = SC_1((As+As')/2, n, K); time_SC = toc;
            ttime_SC = ttime_SC + time_SC; 
        end

        %% Projected Gradient Method (PPM)
        if run_PPM == 1
            H0 = LP_gurobi(H0, n, pi, K);
            opts = struct('T', maxiter, 'tol', tol, 'report_interval', 1, 'quiet', 1);
            tic; H_PPM = PPM(As, H0, pi, opts); time_PPM=toc;
            ttime_PPM = ttime_PPM + time_PPM; H_PPM = sparse(H_PPM);
            X_PPM = H_PPM*H_PPM'; PPM_collector(:,:,iter) = H_PPM;
            fval_PPM(iter) = -trace(H_PPM'*As*H_PPM);
        end
        
        %% Solve the SDP by ADMM
        if run_SDP == 1
            X0 = H0*H0';
            opts = struct('rho', 1e2, 'T', maxiter, 'tol', tol, 'report_interval', 1e2, 'quiet', 1);
            tic; X_SDP = sdp_admm1(As, X0, K, opts); time_SDP=toc;
            ttime_SDP = ttime_SDP + time_SDP;
            SDP_collector(:,:,iter) = X_SDP;
            fval_SDP(iter) = -trace(sparse(X_SDP)'*As);
        end     

end 

color_choice = 8;
subplot(2,3,1); imagesc(Ht*Ht'); title('Ground truth of Polbooks');

%% compute the misclassified vertices of each method
if run_PPM == 1
    [min_PPM, inx] = min(fval_PPM); 
    H_PPM = PPM_collector(:,:,inx); 
%     dist_PPM = 2*n^2/K - 2*norm(H_PPM'*Ht,'fro')^2;
    mis_points_PPM = misclassify_points(H_PPM,Ht);
    subplot(2,3,2); imagesc(H_PPM*H_PPM'); title('PPM'); 
    fprintf('MVs of PPM: %d, time of PPM: %.2f \n', mis_points_PPM, ttime_PPM);
end

if run_MLE == 1
    [min_MLE, inx] = min(fval_MLE); 
    X_MLE = H_MLE*H_MLE'; e_MLE = labelsFromX(X_MLE,K); 
    H_MLE = zeros(n,K); 
    for k = 1:K
       inx = find(e_MLE == k); H_MLE(inx,k) = 1;
    end
%     dist_MLE = 2*n^2/K - 2*norm(H_MLE'*Ht,'fro')^2;   
    mis_points_MLE = misclassify_points(H_MLE,Ht);
    subplot(2,3,3); imagesc(H_MLE*H_MLE'); title('PMLE'); 
    fprintf('MVs of PMLE: %d, time of PMLE: %.2f \n', mis_points_MLE, ttime_MLE);
end

if run_SDP == 1
    [min_SDP, inx] = min(fval_SDP); X_SDP = SDP_collector(:,:,inx);
    e_SDP = labelsFromX(X_SDP,K); 
    H_SDP = zeros(n,K); 
    for k = 1:K
       inx = find(e_SDP == k); H_SDP(inx,k) = 1;
    end
%     dist_SDP = 2*n^2/K - 2*norm(H_SDP'*Ht,'fro')^2;       
    mis_points_SDP = misclassify_points(H_SDP,Ht);
    subplot(2,3,4); imagesc(H_SDP*H_SDP'); title('SDP'); 
    fprintf('MVs of SDP: %d, time of SDP: %.2f \n', mis_points_SDP, ttime_SDP);
end

if run_SC == 1
    H_SC = zeros(n,K);
    for k = 1:K
       inx = find(e_SC == k); H_SC(inx,k) = 1;
    end
%     dist_SC = 2*n^2/K - 2*norm(H_SC'*Ht,'fro')^2;
    mis_points_SC = misclassify_points(H_SC,Ht);
    subplot(2,3,5); imagesc(H_SC*H_SC'); title('SC'); 
    fprintf('MVs of SC: %d, time of SC: %.2f \n', mis_points_SC, ttime_SC);
end

