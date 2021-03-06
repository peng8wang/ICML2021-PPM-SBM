
function H = PMLE_1(A, n, K)

%%%% Algorithm 2 of "Achieving Optimal Misclassification Proportion in SBM" (Gao et al, 2017) %%%% 
    
    inx_full = 1:n;
    
    for u = 1:n
        
        inx = inx_full((inx_full-u)~=0);
        A_u = A(inx,inx);
        H = spectral_init(A_u, n-1, K); % initial label vector generated by spectral clustering
        H1 = zeros(n,K);
        H1(1:u-1,:) = H(1:u-1,:);
        H1(u,:) = zeros(1,K);
        H1(u+1:n,:) = H(u:n-1,:);
        H = H1;
        size_C = sum(H~=0,1);

        %% compute the edges between Ck and Cl
        B = zeros(K);
        for k = 1:K
            for l = 1:K
                num_edge = H(:,k)'*A*H(:,l);
                if l == k
                    B(k,l) = 2*num_edge/(size_C(k)*(size_C(k)-1));
                else
                    B(k,l) = num_edge/(size_C(k)*size_C(l));
                end
            end
        end

        B(B>0.99) = 0.99;
        a = n*min(diag(B)); b = n*max(max(B - diag(diag(B))));
        t = 1/2*( log(a*(1-b/n)) - log(b*(1-a/n)) );
        rho = -1/(2*t)*( log(a*exp(-t)+n-a)-log(b*exp(t)+n-b) );

        sum_l = zeros(K,1);
        for l = 1:K
          inx_l = find(H(:,l));
          sum_l(l) = sum(A(u,inx_l)) - rho*size(inx_l,1); 
        end
        [~,inx] = max(sum_l);
        H(u,inx) = 1; 
       
        H_collect{u} = H;
    end
    %% consensus
    H = zeros(n,K); H1 = H_collect{1};
    H(1,:) = H1(1,:);
    
    for u = 2:n
        count = zeros(K,1);
        H2 = H_collect{u};
        for l = 1:K
            A = find(H1(:,l));
            a = find(H2(u,:));
            B = find(H2(:,a));
            count(l) = size(intersect(A,B),1);
        end
        [~,inx_l] = max(count);
        H(u,inx_l) = 1;
    end
    
end