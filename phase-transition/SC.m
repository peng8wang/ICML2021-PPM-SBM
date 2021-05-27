
function Q = SC(A, n, K)
    
%%%% Algorithm 1 of "Strong Consistency of Spectral Clustering for SBMs" (Su et al., 2019) %%%% 

    % fprintf(' ******************** Spectral Clustering for Exact Recovery *************************** \n')
    %% computing K leading eigenvectors of A
    De = sum(A,2); 
    De = De.^(-1/2); 
    L = diag(De)*A*diag(De); %% compute the graph Laplacian
    [U, D] = eigs(L, K);   
    ev = diag(D);  
    [~,I] = sort(ev,'descend');
    ev = ev(I); 
    U = U(:,I); 
    
    %% Applying K-means to do clustering
    e = kmeans(U, K, 'replicates', 30);  
    Q = zeros(n,K);
    
    for i = 1:K
        Q(e==i,i) = 1;
    end
    
end