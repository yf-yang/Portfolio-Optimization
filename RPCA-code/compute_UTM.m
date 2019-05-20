function [ Sigma_UTM, F_UTM, R_UTM ] = compute_UTM(Sigma_SAM, lambda, N)

M = size( Sigma_SAM, 1 );
[ eigvec, eigval ] = order_eig(Sigma_SAM);
lambda_p = 2 * lambda / N;
for i = 1 : M
    beta = ( i*lambda_p + sum(eigval(i+1:M)) ) / (M-i);
    if beta > eigval(i) - lambda_p
        break;
    end
end
K = i - 1;
beta = ( K*lambda_p + sum(eigval(K+1:M)) ) / (M-K);
F_UTM = eigvec(:,1:K) *  diag( eigval(1:K) - lambda_p ) * eigvec(:,1:K)';
R_UTM = beta * eye(M);
Sigma_UTM = F_UTM + R_UTM;


