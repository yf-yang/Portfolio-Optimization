function [ Sigma_STM, F_STM, R_STM ] = compute_STM( Sigma_SAM, lambda, N )
M = size( Sigma_SAM, 1 );
lambda_p = lambda / (N/2);
init_R =  ones(M, 1);
init_Vhalf = init_R.^(-0.5);
T = init_Vhalf / exp(mean(log(init_Vhalf))); % normalize the transformation matrix
iter_count = 0;
while 1
    T_Sigma = zeros(M, M);
    %% tranform the data
    for d1 = 1 : M
        for d2 = d1 : M
            T_Sigma(d1,d2) = T(d1)*Sigma_SAM(d1,d2)*T(d2);
            T_Sigma(d2,d1) = T_Sigma(d1, d2);
        end
    end
    [ T_eigvec, T_eigval ] = order_eig( T_Sigma); % T means transformed
    iter_count = iter_count + 1;
        
    %% UTM
    sum_eigval = trace( T_Sigma );
    for i = 1 : M
        beta = ( i*lambda_p + sum_eigval - sum(T_eigval(1:i)) ) / (M-i);
        if beta > T_eigval(i) - lambda_p
            break;
        end
    end
    K = i - 1;
    beta = ( K*lambda_p + sum_eigval - sum(T_eigval(1:K)) ) / (M-K);
    T_eigval = [ T_eigval(1:K) - lambda_p  ];
    
    %% evaluate inverse of hat{Sigma_SAM}; ie., vI-G
    inv_hat_Sigma = eye(M,M) / beta;
    for d =1 : K
        inv_hat_Sigma = inv_hat_Sigma - T_eigvec(:,d) * ( 1/beta - 1/T_eigval(d) ) * T_eigvec(:,d)';
    end
    
    %% update T
    new_T = solve_T( T, inv_hat_Sigma.*Sigma_SAM, 1);
    if max(abs((new_T./T)-1))<0.001
        break;
    else
        T = new_T;
    end   
end
F_STM = diag(T.^-1) * T_eigvec(:,1:K) * diag(T_eigval-beta) * T_eigvec(:,1:K)' * diag(T.^-1);
R_STM = diag(T.^-1) * beta * eye(M) * diag(T.^-1);
Sigma_STM = F_STM + R_STM;
