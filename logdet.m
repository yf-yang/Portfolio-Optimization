function ld = logdet( Sigma )    
[ L, p] = chol( Sigma );
if p~= 0 % not PSD
    ld = -inf;
else
    ld = sum(log(diag(L)))*2;
end
