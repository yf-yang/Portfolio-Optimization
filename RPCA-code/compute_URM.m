function [ Sigma_URM, F_URM, R_URM ] = compute_URM( Sigma_SAM, K )

M = size( Sigma_SAM, 1 );
[ eigvec, eigval ] = order_eig(Sigma_SAM);
sigma2_URM = mean( eigval(K+1:M) );
F_URM = eigvec(:, 1:K) * diag( eigval(1:K) - sigma2_URM ) * eigvec(:, 1:K )';
R_URM = sigma2_URM * eye(M);
Sigma_URM = F_URM + R_URM;
