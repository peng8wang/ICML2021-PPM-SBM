
function X = projSp(X0, r, tol)
    if nargin < 2
%      [U,D] = eigs(0.5*(X0+X0'), round(0.5*size(X0,1)),'LA');
        [U,D] = eig(symmetrize(X0));
    else
%         opts = struct('issym',1);
        if nargin > 2
            opts.tol = tol;
        end
%         opts
        [U,D] = eigs(symmetrize(X0), r, 'lm', opts);
    end
    
    idx = diag(D) >= 0;
    X = U(:,idx)*D(idx,idx)*U(:,idx)';
end

function Z = symmetrize(X)
    Z = 0.5*(X+X');
end
