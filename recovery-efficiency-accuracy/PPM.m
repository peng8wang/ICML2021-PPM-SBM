
function [H, iter, fval_collector] = PPM(A, H0, pi, opts)

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
        [n, K] = size(H0); fval_collector=[];  
        % fprintf(' ******************** PGM Method for Community Recovery *************************** \n')
        
        %% constraints parameters for LP
        H = H0; 
        AH = A*H; fval = -trace(H'*AH);
                
        for iter = 1:maxiter
                
                fval_old = fval;
                %% compute the projection via MCAP or linear programming solver in GUROBI 
                if K == 2
                    diff = AH(:,1) - AH(:,2); 
                    [~,inx] = sort(diff); 
                    G = zeros(n,K);
                    G(inx(pi(2)+1:n),1) = 1; G(inx(1:pi(2)),2) = 1;
                else
                    G = LP_gurobi(AH, n, pi, K);
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
                fval_collector(iter) = fval; flag = 0;
                H_collector(:,:,iter) = H; 
                
                if iter > 5
                    for i = 1:5
                        if norm(H - H_collector(:,:,iter-i), 'fro') < tol
                            flag = 1;
                            break;
                        end
                    end
                end
                
                if abs(gap) <= tol || flag
                        break;
                end                

        end
        

end