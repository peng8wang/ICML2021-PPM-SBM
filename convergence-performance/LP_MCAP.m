
function H = LP_MCAP(C,n,K)
    
    %% implement the "Geometric Algorithm for the MCAP" (Tokuyama & Nakano, 1995) %%
    C = -C;
    C = C - mean(C,2); % each row denote a point p(u_i) for i = 1,...,n
    G = sum(C,1)'/n;
    mu = zeros(K,1); 
    m = n/K;
    
    %% compute the difference of row  
    diff_C_column = zeros(n,K*K);
    for k = 1:K
        for j = 1:K
            diff_C_column(:,(k-1)*K+j) = C(:,k) - C(:,j);
        end
    end
    
    %% generate the set of p(i,j), i~=j
    [~,set_M_label] = min(C(1:m,:)'-G);
    P = zeros(K*K,K); P_inx = zeros(K*K,1); P_i_j = zeros(K*K,1);
    for k = 1:K
       tilde_Si = find(set_M_label == k); 
       mu(k) = sum(set_M_label == k);
       if mu(k) > 0
           for j = 1:K     
               inx = (k-1)*K+j;
               diff_i_j = diff_C_column(tilde_Si,inx);
               [diff_p_i_j,inx_p_i_j] = max(diff_i_j);
               P(inx,:) = C(tilde_Si(inx_p_i_j),:); 
               P_i_j(inx) = diff_p_i_j;
               P_inx(inx) = tilde_Si(inx_p_i_j);
           end   
       end
    end

    %% insert new point and update the splitter iteratively
    label_vector = zeros(n,1); label_vector(1:m) = set_M_label; K_set = 1:K;

    for i = m+1:n

        %% add the new point i into region T(G;a)
        [~,a] = min(C(i,:)' - G);         
        label_vector(i) = a; 
        mu(a) = mu(a) + 1; 
        
        %% update the set P
        set_M_label = label_vector(1:i);
        for k = a
           tilde_Si = find(set_M_label == k); 
           if mu(k) > 0
               for j = 1:K     
                   inx = (k-1)*K+j;
                   diff_i_j = diff_C_column(tilde_Si,inx);
                   [diff_p_i_j,inx_p_i_j] = max(diff_i_j);
                   P(inx,:) = C(tilde_Si(inx_p_i_j),:); 
                   P_i_j(inx) = diff_p_i_j;
                   P_inx(inx) = tilde_Si(inx_p_i_j);
               end  
           end
        end
        
        %% sliding procedure
        if mu(a) >= m + 1
            inx_J = a; num_boundary = 0;
            boundary = []; boundary_inx = []; boundary_s_h = [];

            while true
                e_J = zeros(K,1); e_J(inx_J) = 1;
                X_J = e_J - sum(e_J)/K*ones(K,1);
                inx_K_J = K_set(~ismember(K_set,inx_J));

                %% find g(i,j) such that i \in J, j \in K-J
                size_J = size(inx_J,2); size_K_J = size(inx_K_J,2); 
                g_i_j = zeros(size_J*size_K_J,K); % set of g(i,j)
                diff = zeros(size_J*size_K_J,1);  % set of lambda for each g(i,j)
                set_g_i_j = zeros(size_J*size_K_J,2); % set of index of g(i,j)
                k = 1; 
                for l = inx_J
                   for j = inx_K_J
                       %% g(i,j) is the intesection of x_i-x_j = p(i,j)_i - p(i,j)_j and l(J)
                       diff(k) = P_i_j((l-1)*K+j)-(G(l)-G(j));
                       set_g_i_j(k,:) = [l,j];
                       g_i_j(k,:) = G + diff(k)*X_J;
                       k = k + 1;
                   end           
                end
                %% update G
                [~,inx] = max(diff);
                G = g_i_j(inx,:)'; s = set_g_i_j(inx,1); h = set_g_i_j(inx,2);
                p_s_h = P((s-1)*K+h,:); inx_p_s_h = P_inx((s-1)*K+h);

                if mu(h) < m      
                    label_vector(inx_p_s_h) = h; 
                    mu(h) = mu(h) + 1;
                    mu(s) = mu(s) - 1;
                    %% assign the boundary elements suitably                   
                    if num_boundary >= 1 && sum(mu > m) > 0
                            T = 1; 
                            inx_a = find(boundary_s_h(:,1)==a)';
                            inx_collect=[]; inx_collect{T} = inx_a;
                            while true
                                T = T + 1; inx_T = [];
                                for k = inx_a                                 
                                    inx_T = [inx_T,find(boundary_s_h(:,1)==boundary_s_h(k,2))'];
                                end 
                                inx_collect{T} = inx_T;
                                label_vectors = cell(T,T); mus = cell(T,T);
                                label_vectors{1} = label_vector; mus{1} = mu;
                                for t = 1:T
                                    t1 = 1;                                   
                                    for l = inx_collect{t}
                                        [mu2, label_vector2] = label_exchange(cat(1,label_vectors{t,:}),...
                                            cat(1,mus{t,:}), boundary_inx, boundary_s_h, l, K, inx_collect, t, t1);
                                        if sum(mu2 <= m) == K
                                            mu = mu2; label_vector = label_vector2;
                                            break; 
                                        else
                                            label_vectors{t+1,t1} = label_vector2; 
                                            mus{t+1,t1} = mu2;
                                            t1 = t1 + 1;
                                        end   
                                    end    
                                    if sum(mu <= m) == K
                                        break; 
                                    end  
                                end    
                                if sum(mu <= m) == K
                                    break; 
                                else
                                    inx_a = inx_collect{T};
                                end                    
                            end
                    end                     
                    break;                   
                else
                    inx_J = [inx_J,h];
                    num_boundary = num_boundary + 1;
                    boundary(num_boundary,:) = p_s_h;
                    boundary_inx(num_boundary) = inx_p_s_h;
                    boundary_s_h(num_boundary,:) = [s,h];
                end
            end
            
            %% update the set P
            set_M_label = label_vector(1:i);
            for k = 1:K
                tilde_Si = find(set_M_label == k); 
                if mu(k) > 0
                    for j = 1:K     
                       inx = (k-1)*K+j;
                       diff_i_j = diff_C_column(tilde_Si,inx);
                       [diff_p_i_j,inx_p_i_j] = max(diff_i_j);
                       P(inx,:) = C(tilde_Si(inx_p_i_j),:); 
                       P_i_j(inx) = diff_p_i_j;
                       P_inx(inx) = tilde_Si(inx_p_i_j);
                    end  
                end
            end
            
        end

    end

    H = zeros(n,K);
    for k = 1:K
       H(label_vector == k, k) = 1;
    end
    
end

%% assign the boundary elements suitably  
function [mu,label_vector,h] = label_exchange(label_vector, mu, boundary_inx, boundary_s_h, l, K, inx_collect, T, T1)
    
    k = size(mu,1)/K; n = size(label_vector,1)/k; m =n/K;
    flag = 0;

    for t = 1:k
        mu1 = mu((t-1)*K+1:t*K);
        label_vector1 = label_vector((t-1)*n+1:t*n);
        inx = boundary_inx(l);
        s = boundary_s_h(l,1);
        h = boundary_s_h(l,2);
        mu1(s) = mu1(s) - 1;
        label_vector1(inx) = h;
        mu1(h) = mu1(h) + 1;

        if sum(mu1<=m) == K
            mu = mu1;
            label_vector = label_vector1;
            break;      
        else
            flag = flag + 1;
        end
        mus{t} = mu1;
        label_vectors{t} = label_vector1;        
    end
    
    if flag == k
        if T > 1
            s_h_T = boundary_s_h(inx_collect{T}(T1),:);
            for t = 1:k
                s_h = boundary_s_h(inx_collect{T-1}(t),:);
                if s_h(2) == s_h_T(1)
                    break;
                end
            end
        end
        mu = mus{t};
        label_vector = label_vectors{t};
    end

end

