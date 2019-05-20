function [ Sigma_EM, F_EM, R_EM ] = compute_EM( Sigma_SAM, K)

M = size( Sigma_SAM, 1 );
[ eigvec, eigval ] = order_eig(Sigma_SAM);
Fhalf = zeros(M, K);
sigma2 = mean( eigval(K+1:M) );
for i = 1 : K
    Fhalf(:,i) = sqrt(eigval(i)-sigma2)*eigvec(:,i);
end
R = diag(Sigma_SAM);
for i = 1 : K
    R = R - Fhalf(:,i).^2;
end

if K == 0
    Sigma_EM = diag(R);
    F_EM = zeros(M,M);
    R_EM = diag(R);
    return;
end

iter = 0;
while 1
    big = inv( eye(K) + Fhalf' * diag(R.^-1) * Fhalf );
    term1 = zeros(M, K);
    term2 = zeros(K, K);
   
    temp = big * Fhalf' * diag(R.^-1);
    term1 = Sigma_SAM * temp' ;
    term2 = big + temp * Sigma_SAM * temp'; 
    
    newFhalf = term1 * inv(term2);
    newR = diag(Sigma_SAM) - diag(newFhalf * term1');
    if max( abs(R-newR)./R )<0.001
        break;
    else
        R = newR;
        Fhalf = newFhalf;
    end
    iter = iter + 1;
end
R_EM = diag(R);
F_EM = Fhalf * Fhalf';
Sigma_EM = F_EM + R_EM;
