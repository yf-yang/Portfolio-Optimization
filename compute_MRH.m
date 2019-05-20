function [ Sigma_MRH, F_MRH, R_MRH ] = compute_MRH( Sigma_SAM, K)

M = size( Sigma_SAM, 1 );
[ eigvec, eigval ] = order_eig(Sigma_SAM);
sigma2 = mean( eigval(K+1:M) );

F_MRH = eigvec(:,1:K) * diag(eigval(1:K)-sigma2) * eigvec(:,1:K)';
R_MRH = diag( diag(Sigma_SAM)-diag(F_MRH) );
Sigma_MRH = F_MRH + R_MRH;
