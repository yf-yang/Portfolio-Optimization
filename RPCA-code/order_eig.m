function [ sorted_eigvec, sorted_eigval ] = order_eig( Sigma ) %% sort in descending order

M = size( Sigma, 1);
Sigma = (Sigma + Sigma')/2; %% to remove rounding error that destroy symmetry
[ orig_eigvec, orig_eigval ] = eig(Sigma);
orig_eigval = real( diag( orig_eigval ) );
[ sorted_eigval, index ] = sort( orig_eigval, 'descend');
sorted_eigvec = zeros( M, M);
for i = 1 : M
    sorted_eigvec(:,i) = orig_eigvec( :, index(i) );
end