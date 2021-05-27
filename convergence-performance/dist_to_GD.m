
function dist = dist_to_GD(H, Ht)

    %% compute the distance to ground truth min ||H-HtQ||_F s.t. Q \in \Pi_K.

    [n,K] = size(H);
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
    
    dist = sqrt(2*n - 2*trace(H'*Ht*Q));


end