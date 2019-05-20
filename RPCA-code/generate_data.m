function [ X, Sigma_s, F_s, R_s ] = generate_data( M, K, N, uniform, sigma_f, sigma_r )

%% generate basis ( no effect if isotropic noise is assumed )
Psi = randn( M, K);
for d1 = 1 : K
    for d2 = 1:d1-1
        Psi(:,d1) = Psi(:,d1) - ( Psi(:,d2)'*Psi(:,d1) )*Psi(:,d2);
    end
    Psi(:,d1) = Psi(:,d1)/norm(Psi(:,d1));
end

%% generate factor coefficients
f = randn( K, 1) * sigma_f;
f = sort( abs(f), 'descend' );


%% generate residuals (uniform / inverse-gamma)
if uniform == 1
    R_s = ones(M ,1);
else
    %% inverse gamma
    R_s = exp( randn(M, 1) * sigma_r );
end
R_sqrt = sqrt(R_s);
R_s = diag(R_s);

%% generate Sigma_s
Fhalf = zeros( M, K);
for d = 1 : K
    Fhalf(:,d) = f(d)*Psi(:,d);
end
F_s = Fhalf * Fhalf';
Sigma_s = F_s + R_s;

%% generate data
X = zeros(M, N);
for n = 1 : N
    zn = randn( K, 1);
    wn = randn( M, 1);
    X(:,n) = Fhalf * zn + wn.*R_sqrt;
end
