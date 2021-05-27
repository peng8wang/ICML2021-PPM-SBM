
function e = SC_1(A, n, K)
    
%%%% Algorithm 1 of "Strong Consistency of Spectral Clustering for SBMs" (Su et al., 2019) %%%% 

    % fprintf(' ******************** Spectral Clustering for Exact Recovery *************************** \n')
    
    %% compute K leading eigenvectors of graph Laplician
    De = sum(A,2); 
    De = De.^(-1/2); L = diag(De)*A*diag(De);
    [U, D] = eigs(L, K);   
    ev = diag(D);  
    [~,I] = sort(ev,'descend');
    ev = ev(I);        
    U = U(:,I); U = normr(U);
    
    %% apply K-means to do clustering
    e = kmeans(U, K, 'replicates', 50);  
    
%     Q = zeros(n,K);
%     
%     for i = 1:K
%         inx = find(e==i);
%         Q(inx,i) = 1;
%     end
    
end