
function e = SC(A, n, K)
    
%%%% Algorithm 1 of "Consistency of Spectral Clustering in SBMs" (Lei & Rinaldo, 2015) %%%% 

    % fprintf(' ******************** Spectral Clustering for Exact Recovery *************************** \n')
    %% computing K leading eigenvectors of graph Laplician
    [U, D] = eigs(A, K);   
    ev = diag(D);  
    [~,I] = sort(ev,'descend');
    ev = ev(I);        
    U = U(:,I); U = normr(U);
    
    %% Applying K-means to do clustering
    e = kmeans(U, K, 'replicates', 50);  
%     Q = zeros(n,K);
%     
%     for i = 1:K
%         inx = find(e==i);
%         Q(inx,i) = 1;
%     end
    
end