
function H = LP_gurobi(C,n,K)

    %% solve the MCAP by Gurobi solver %% 
    m = n/K;
    %% constraints parameters for LP
    In = sparse(eye(n)); Ik = sparse(eye(K));
    Aeq = repmat(In,1,K);
    Aeq1 = kron(Ik,ones(n,1))';

    %% solve the prime LP via gurobi 
    model.obj = reshape(C,[],1); 
    model.modelsense = 'Max';
    model.A = [Aeq; Aeq1];
    model.rhs = [ones(n,1); m*ones(K,1)];
    model.sense = [repmat('=',n,1); repmat('=',K,1)];        
    model.lb = zeros(n*K,1); 
    params.OutputFlag = 0;
    result = gurobi(model, params);
    H = reshape(result.x,n,K);

    
end
    