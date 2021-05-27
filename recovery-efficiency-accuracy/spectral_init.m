
function H = spectral_init(As, n, K)
    
%%%% Algorithm 2 of "Achieving Optimal Misclassification Proportion in SBM" (Gao et al, 2017) %%%% 
    
    H = zeros(n,K);              % define clustering matrix
    mu = 1e-2; r = mu*sqrt(K/n); % define critical radius
   
    %% computing distance between rows of K leading eigenvectors    
    [U, ~] = eigs(As, K);
    M = U*U'; M1 = repmat(diag(M),1,n);
    dist = sqrt(M1 + M1' - 2*M);                  
    Q = sparse(dist < r); S = 1:n;
    
    %% greedy method to do clustering    
    for k = 1:K
        
        %% compute tk
        Q_S = Q(S,:);
        num_ele = sum(Q_S~=0,2);
        [~,t] = max(num_ele);
        
        %% compute Ck
        T = find(Q(S(t),:));    
        C{k} = T;
        H(T,k) = 1;
        
        %% S = S\Ck
        S = S(~ismember(S,T));
        
    end
    
    %% clustering by distance
    if isempty(S) == 0
        for i = S
           dist_i = zeros(K,1);
           for k = 1:K
                T = C{k};
                dist_i(k) = sum(dist(i,T))/size(T,2);
           end
           [~,inx_k] = min(dist_i);
           H(i,inx_k) = 1;
        end
    end
    
    %% refine by kmeans
    H = normr(H);
    e = kmeans(H, K, 'replicates', 40); 
    H = zeros(n,K);
    for i = 1:K
        H(e==i,i) = 1;
    end
    

end

        
%        size_S = size(S,2);
%        num_ele = zeros(size_S,1);                    
%         for i = 1:size_S
%            A = find(Q(S(i),:));              
%            num_ele(i) = size(A,2); 
%         end