
function l = misclassify_points(H, Ht)


    [n, K] = size(H);
    H = sparse(H); Ht = sparse(Ht);    
    
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
    
    D = full(H - Ht*Q);
    l = sum(vecnorm(D,2,2) > 1e-8);

end