function [ X, Sigma_s, c, F_s, R_s ] = generate_data( M, N, uniform, objective, mu_f, sigma_f, sigma_p, sigma_r )

%% generate basis ( no effect if isotropic noise is assumed )
Psi = randn(M, M);
for d1 = 1 : M
    for d2 = 1:d1-1
        Psi(:,d1) = Psi(:,d1) - ( Psi(:,d2)'*Psi(:,d1) )*Psi(:,d2);
    end
    Psi(:,d1) = Psi(:,d1)/norm(Psi(:,d1));
end

%% generate factor coefficients
f = randn(M, 1) * sigma_f + mu_f;
f = exp(f);
f = sort( abs(f), 'descend' );

%% generate c
c = randn(M, 1);
if objective == 1
    p = randn(20, 1) * sigma_p;
    for d = 1 : 20
        c = c + p(d)*Psi(:,d);
    end
end
c = c / norm(c);

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
Fhalf = zeros( M, M );
for d = 1 : M
    Fhalf(:,d) = f(d)*Psi(:,d);
end
F_s = Fhalf * Fhalf';
Sigma_s = F_s + R_s;

%% generate data
X = zeros(M, N);
for n = 1 : N
    zn = randn( M, 1);
    wn = randn( M, 1);
    X(:,n) = Fhalf * zn + wn.*R_sqrt;
end
