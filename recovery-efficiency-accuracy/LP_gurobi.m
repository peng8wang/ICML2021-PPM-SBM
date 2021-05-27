
function H = LP_gurobi(G, n, pi, K)

    %% solve the LP: max <G,H> s.t. H1_K = 1_n, H1_n = n/K 1_K, H \in {0,1} %%

    %% constraints parameters for LP
    In = sparse(eye(n)); Ik = sparse(eye(K));
    Aeq = repmat(In,1,K);
    Aeq1 = kron(Ik,ones(n,1))'; 
    
    %% solve the subproblem by Gurobi                
    model.obj = reshape(G,[],1); 
    model.modelsense = 'Max';
    model.A = [Aeq; Aeq1];
    model.rhs = [ones(n,1); pi];
    model.sense = [repmat('=',n,1); repmat('=',K,1)];
    model.lb = zeros(n*K,1); 
    params.OutputFlag = 0;
    result = gurobi(model, params);
    H = reshape(result.x,n,K);
    
end