
clear all; clc;

%% basic setting of parameters
n = 300;      %%%  n = the number of nodes
K = 3;        %%%  K = the number of communities
m = n/K;      %%%  m = the community size
nnt = 10;     %%%  the number of repeating the trials for fixed alpha, beta

%% ground truth 
Xt =  kron(sparse(eye(K)), ones(m)); 

%% set the ranges of alpha, beta
if K == 3 %%% the case of K = 3
    max_a = 30; max_b = 10; 
    arange = 0:0.5:max_a; brange = 0:0.4:max_b; 
else      %%% the case of K = 6
    max_a = 60; max_b = 20; 
    arange = 0:1:max_a; brange = 0:0.8:max_b; 
end
nna = length(arange); nnb = length(brange);

%% record information
[prob_SDP, prob_MLE, prob_SC, prob_PPM]  = deal(zeros(nna,nnb));  %%% record ratio of exact recovery
[ttime_PPM, ttime_MLE, ttime_SC, ttime_SDP] = deal(0);            %%% record total running time

%% choose the running algorithm
run_SDP = 0; run_MLE = 1; run_SC = 1; run_PPM = 1;

for iter1 = 1:nna      %%% choose alpha
    
    a = arange(iter1); 
    
    for iter2 = 1:nnb  %%% choose beta 
            
        b = brange(iter2); 
        p = a*log(n)/n; q = b*log(n)/n; %%% p: the inner connecting probability; q: the outer connecting probability;           
        [succ_SDP, succ_MLE, succ_SC, succ_PPM] = deal(0);

        for iter3 = 1:nnt %%% repeating the trials 
                
                %% generate an adjacency matrix A by the symmtric SBM
                A = sparse(zeros(n));
                for i = 1:K
                  for j = i:K
                      if i == j
                         Am1 = rand(m); Am1 = tril(Am1,-1);                         
                         Am1 = Am1' + Am1 + diag(diag(Am1));
                         Am = sparse(Am1 < p);
                         A((i-1)*m+1:i*m,(j-1)*m+1:j*m) = Am;
                      else
                         Am1 = rand(m); Am = sparse(Am1 < q);                         
                         A((i-1)*m+1:i*m,(j-1)*m+1:j*m) = Am;
                         A((j-1)*m+1:j*m,(i-1)*m+1:i*m) = Am'; 
                      end
                  end
                end

                %% choose the parameters
                H = randn(n,K); Q0 = normr(H);
                tol = 1e-5; report_interval = 5; total_time = 1e3;
                
                %% PPM for the MLE
                if run_PPM == 1                    
                    opts = struct('T', 2, 'tol', tol, 'report_interval', report_interval,...
                        'total_time', total_time, 'solver', 'MCAP');
                    tic; [H_PPM, iter_PPM, fval_collector_PPM] = PPM(A, H, opts); time_PPM=toc;
                    ttime_PPM = ttime_PPM + time_PPM; H_PPM = sparse(H_PPM);
                    dist_PPM =  sqrt(2*n^2/K-2*trace(H_PPM'*Xt*H_PPM));
                    if dist_PPM <= 1e-3
                        succ_PPM = succ_PPM + 1;
                    end
                end

                %% Penalized MLE based method
                if run_MLE == 1                               
                    tic; H_MLE = PMLE(A, n, K); time_MLE=toc;
                    ttime_MLE = ttime_MLE + time_MLE; H_MLE = sparse(H_MLE);
                    dist_MLE =  sqrt(2*n^2/K-2*trace(H_MLE'*Xt*H_MLE));
                    if dist_MLE <= 1e-3
                        succ_MLE = succ_MLE + 1;
                    end
                end

                %% ADMM for SDP
                if run_SDP == 1 && sqrt(a) - sqrt(b) >= sqrt(K-1.5)
                    opts = struct('rho', 1e-1, 'T', 1e2, 'tol', 1e-1, 'quiet', 1, ...
                            'report_interval', report_interval, 'total_time', total_time, 'r', n*2/3);
                    tic; [X_SDP, fval_collector_SDP] = sdp_admm1(A, Xt, K, opts); time_SDP = toc;
                    ttime_SDP = ttime_SDP + time_SDP;
                    X_SDP(X_SDP >= 0.5) = 1; X_SDP(X_SDP < 0.5) = 0; X_SDP = sparse(X_SDP);
                    dist_SDP = sqrt(2*n^2/K-2*trace(X_SDP'*Xt));
                    if dist_SDP <= 1e-1
                        succ_SDP = succ_SDP + 1;
                    end
                end

                %% Spectral clustering
                if run_SC == 1
                    tic; H_SC = SC(A, n, K); time_SC = toc;
                    ttime_SC = ttime_SC + time_SC; H_SC = sparse(H_SC);
                    dist_SC =  sqrt(2*n^2/K-2*trace(H_SC'*Xt*H_SC));
                    if dist_SC <= 1e-3
                        succ_SC = succ_SC + 1;
                    end
                end

                fprintf('Outer iter: %d, Inner iter: %d,  Repated Num: %d \n', iter1, iter2, iter3);
        end

        prob_PPM(iter1, iter2) = succ_PPM/nnt;
        prob_SDP(iter1, iter2) = succ_SDP/nnt;
        prob_MLE(iter1, iter2) = succ_MLE/nnt;
        prob_SC(iter1, iter2) = succ_SC/nnt;
            
    end    
end 

%% Plot the Phase Transition
f =  @(x,y)  sqrt(y) - sqrt(x) - sqrt(K); 

if run_PPM == 1
    figure(); imshow(prob_PPM(2:61,2:26), 'InitialMagnification','fit','XData',[0 max_b],'YData',[0 max_a]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 max_b 0 max_a], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]); set(gca,'FontSize', 12, 'FontWeight','bold');    
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('PPM');
end

if run_SC == 1
    figure();
    imshow(prob_SC(2:61,2:26), 'InitialMagnification','fit','XData',[0 max_b],'YData',[0 max_a]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 max_b 0 max_a], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]); set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('SC');
end

if run_SDP == 1
    figure();
    imshow(prob_SDP(2:61,2:26), 'InitialMagnification','fit', 'XData',[0 max_b],'YData',[0 max_a]); colorbar;
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 max_b 0 max_a], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]); set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('SDP');
end

if run_MLE == 1
    figure(); 
    imshow(prob_MLE(2:61,2:26), 'InitialMagnification','fit','XData',[0 max_b],'YData',[0 max_a]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 max_b 0 max_a], 'LineWidth', 1.5, 'color', 'r'); 
    daspect([1 3 1]); set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('PMLE');
end

