
function [H, iter, fval_collector] = PPM(A, H0, opts)

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
        [n, K] = size(H0); fval_collector=[];  
        % fprintf(' ******************** PPM Algorithm for Community Estimation *************************** \n')
        
        %% initialize H0 via spectral method
        H = spectral_init(A, n, K);
        AH = A*H; N = ceil(log(log(n))) + ceil(2*log(n)/log(log(n))) + 2;
        
        for iter = 1:maxiter
                                      
                %% compute the projection via MCAP or linear programming solver in GUROBI
                if strcmp(solver, 'MCAP')
                    G = LP_MCAP(AH, n, K);
                else
                    G = LP_gurobi(AH, n, K);
                end

                %% compute the optimality gap                 
                AG = A*G; Dt = G - H; 
                gap = trace(AH'*Dt); 
                
                %% update the iterate                
                H = G; AH = AG;
                fval = -trace(H'*AH);
            
                if mod(iter,report_interval) == 0 && ~quiet            
                    fprintf('iternum: %2d, FW gap: %2.4e, fval: %.2f\n', iter, gap, fval) 
                end
                
                %% record information and check the termination criterion
                fval_collector(iter) = fval;
                
                if abs(gap) <= tol || iter >= N
                        break;
                end                

        end
        

end