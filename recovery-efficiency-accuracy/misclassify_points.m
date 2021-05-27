
function l = misclassify_points(H, Ht)

    %% compute the number of misclassified vertices of an iterate compared to the ground truth    
    [n, K] = size(H);
    H = sparse(H); Ht = sparse(Ht);    
    
    %% solve the problem to get Q = argmin ||H-HtQ||_F s.t. Q \in \Pi_K.
    cvx_solver mosek
    cvx_begin quiet
    cvx_precision high
        variable Q(K,K) 
        maximize trace(H'*Ht*Q)
        subject to
            Q'*ones(K,1) == 1
            Q*ones(K,1) == 1
            Q >= 0
    cvx_end
    
    %% compute the number of misclassified vertices
    D = full(H - Ht*Q);
    l = sum(vecnorm(D,2,2) > 1e-8);

end