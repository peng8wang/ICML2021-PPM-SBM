
%% set the path of gurobi for solving LP of PPM
addpath('D:\Software\gurobi903\win64\matlab');
clear; clc;

%% Load the Real-World Dataset
load Datasets/football        %%% load the set football
As = Problem.A; As(As>0) = 1; %%% As: adjacency matrix
xt = Problem.aux.nodevalue;  xt = xt + 1; %%% xt: ground truth
K = max(xt); pi = []; inx_set = []; 
t = 1; 

%% determine the number of communities and each community size
for k = 1:K
    inx1 = find(xt==k);
    if size(inx1,1) >= 10
        inx{k} = inx1;
        inx_set = [inx_set; inx1];
        pi(t) = size(inx{k},1);
        t = t + 1;
    end
end
As = As(inx_set, inx_set); xt = xt(inx_set);
n = size(As,1); 
K = t-1; pi = pi'; Ht = zeros(n,K); 

%% form the clustering matrix of the ground truth
for k = 1:K
    if k == 1
        inx = 1:pi(k);
    else
        inx = sum(pi(1:k-1))+1:sum(pi(1:k));
    end
   Ht(inx,k) = 1;    
end

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
        H0 = randn(n,K); 
        maxiter = 1e3; tol = 1e-3; report_interval = 1e0; total_time = 1e3;
        
        %% Penalized MLE based method (PMLE)
        if run_MLE == 1
            tic; H_MLE = PMLE(As, n, K); time_MLE=toc;
            ttime_MLE = ttime_MLE + time_MLE; H_MLE = sparse(H_MLE);
            X_MLE = H_MLE*H_MLE'; MLE_collector(:,:,iter) = H_MLE;
            fval_MLE(iter) = -trace(H_MLE'*As*H_MLE);
        end
                
        %% Spectral Clustering
        if run_SC == 1
            tic; e_SC = SC_1((As+As')/2, n, K); time_SC = toc;
            ttime_SC = ttime_SC + time_SC; 
        end

        %% Projected Power Method (PPM)
        if run_PPM == 1
            H0 = LP_gurobi(H0, n, pi, K);
            opts = struct('T', maxiter, 'tol', tol, 'report_interval', report_interval, 'quiet', 1);
            tic; H_PPM = PPM(As, H0, pi, opts); time_PPM=toc;
            ttime_PPM = ttime_PPM + time_PPM; H_PPM = sparse(H_PPM);
            X_PPM = H_PPM*H_PPM'; PPM_collector(:,:,iter) = H_PPM;
            fval_PPM(iter) = -trace(H_PPM'*As*H_PPM);
        end
        
        %% Solve the SDP by ADMM
        if run_SDP == 1
            X0 = H0*H0';
            opts = struct('rho', 1e0, 'T', maxiter, 'tol', tol, 'report_interval', 1e2, 'quiet', 1);
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
    X_PPM = H_PPM*H_PPM'; e_PPM = labelsFromX(X_PPM,K);
    H_PPM = zeros(n,K);
    for k = 1:K
       inx = find(e_PPM == k);
       H_PPM(inx,k) = 1;
    end
%     dist_PPM = 2*n^2/K - 2*norm(H_PPM'*Ht,'fro')^2;
    mis_points_PPM = misclassify_points(H_PPM,Ht); %% misclassified vertices of PPM
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
    mis_points_MLE = misclassify_points(H_MLE,Ht); %% misclassified vertices of PMLE
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
    mis_points_SDP = misclassify_points(H_SDP,Ht); %% misclassified vertices of SDP
    subplot(2,3,4); imagesc(H_SDP*H_SDP'); title('SDP'); 
    fprintf('MVs of SDP: %d, time of SDP: %.2f \n', mis_points_SDP, ttime_SDP);
end

if run_SC == 1
    H_SC = zeros(n,K);
    for k = 1:K
       inx = find(e_SC == k); H_SC(inx,k) = 1;       
    end
%     dist_SC = 2*n^2/K - 2*norm(H_SC'*Ht,'fro')^2;
    mis_points_SC = misclassify_points(H_SC,Ht); %% misclassified vertices of SC
    subplot(2,3,5); imagesc(H_SC*H_SC'); title('SC'); 
    fprintf('MVs of SC: %d, time of SC: %.2f \n', mis_points_SC, ttime_SC);
end



    