
function [H, iter, fval_collector, iter_gap] = PPM(A, H, Ht, opts)

        %% Parameter setting
        maxiter = opts.T;
        tol = opts.tol;
        report_interval = 100;
        quiet = false;
        if isfield(opts,'report_interval') 
            report_interval = opts.report_interval;
        end
        if isfield(opts,'quiet')
            quiet = opts.quiet;
        end
        if isfield(opts, 'solver')
            solver = opts.solver;
        else
            solver = 'MCAP';
        end
        fprintf(' ******************** PPM Method for Community Recovery *************************** \n')       
        
        [n, K] = size(H); AH = A*H;
        fval_collector(1) = trace(H'*AH); 
        iter_gap(1) = dist_to_GD(H,Ht); % sqrt(2*n^2/K-2*trace(H'*Xt*H));
        
        for iter = 1:maxiter                                      
                
                H_old = H;
                
                %% compute the projection via MCAP or linear programming solver in GUROBI 
                if strcmp(solver, 'MCAP')
                    G = LP_MCAP(AH, n, K);
                else
                    G = LP_gurobi(AH, n, K);
                end
                
                %% update the parameter                 
                AG = A*G; Dt = G - H; 
                gap = trace(AH'*Dt);  
                
                %% Update the iterate                
                H = G; AH = AG;
                fval = -trace(H'*AH);
            
                if mod(iter,report_interval) == 0 && ~quiet            
                    fprintf('iternum: %2d, FW gap: %2.4e, fval: %.2f\n', iter, gap, fval) 
                end
                
                %% record information and check the termination criterion
                fval_collector(iter+1) = fval; 
                iter_gap(iter+1) = dist_to_GD(H,Ht); % sqrt(2*n^2/K-2*trace(H'*Xt*H)); 
                               
                if abs(gap) <= tol && norm(H-H_old,'fro') <= tol
                    break;
                end                

        end
        

end