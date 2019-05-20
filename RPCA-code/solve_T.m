function T = solve_T( init_T, A, normalize )
scale_a = 0.2;
scale_b = 0.5;
T = init_T;
value = T'*A*T - 2*sum(log(T));
epsilon = 0.0001;
while 1    
    gradient = 2*A*T - 2*(T.^-1);
    hessian = 2*A + 2*diag(T.^-2);
    dt = - hessian \ gradient;
    if - gradient' * dt / 2 <= epsilon
        break;
    end
    %% backtracking line search
    r = 1;
    next_t = T + r * dt;
    if min(next_t) > 0
        next_value = next_t' * A * next_t - 2 * sum(log(next_t));
    else
        next_value = inf;
    end
    while next_value > value + scale_a * r * gradient'*dt
        r = r * scale_b;
        next_t = T + r * dt;
        if min(next_t) > 0
            next_value = next_t' * A * next_t - 2 * sum(log(next_t));
        else
            next_value = inf;
        end
    end
    T = next_t;
    value = next_value;
end

if normalize == 1
    T = T / exp(mean(log(T)));
end