function [ Sigma_TM, F_TM, R_TM ] = compute_TM( Sigma_SAM, lambda, N )

M = size( Sigma_SAM, 1);
lambda_p = lambda / (N/2);
[ Sigma_UTM, F_UTM, R_UTM ] = compute_UTM(Sigma_SAM, lambda, N);
init_R = diag(Sigma_SAM) - diag(F_UTM);

V = 1./init_R;
Sigma_p = Sigma_SAM - lambda_p*eye(M);
iter_count = 0;
while 1
    Vroot = sqrt(V);
    Sigma_temp = zeros(M, M);
    for d1 = 1 : M
        for d2 = d1 : M
            Sigma_temp(d1,d2) = Sigma_p(d1,d2)*Vroot(d1)*Vroot(d2);
            Sigma_temp(d2,d1) = Sigma_temp(d1, d2);
        end
    end
    [A, S] = order_eig( Sigma_temp );
    iter_count = iter_count + 1;
    T = ones(M,1);
    diag_tilde = zeros(M,1);
    for d =1 : M
        if S(d) > 1
            T(d) = S(d);
            diag_tilde = diag_tilde + A(:,d).^2 * (S(d)-1);
        end
    end
    diag_tilde = diag_tilde + ones(M,1);
    Vnew = diag_tilde./diag(Sigma_SAM);
    if max(abs((Vnew./V)-1))<0.001
        break;
    else
        V = Vnew;
    end   
end
F_TM = diag(1./sqrt(V)) *  A * (diag(T-1)) * A' * diag(1./sqrt(V));
R_TM = diag(1./V);
Sigma_TM = F_TM + R_TM;



