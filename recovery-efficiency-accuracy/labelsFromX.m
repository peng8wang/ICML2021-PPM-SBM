
function [e,U,ev] = labelsFromX(X,K)

    [U,D] = eigs(X,K);
    ev = diag(D);
    [~,I] = sort(ev,'descend');
    ev = ev(I);
    U = U(:,I);
    e = kmeans(U,K,'replicates',30);
    
end