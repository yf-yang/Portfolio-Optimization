clear all;
tic;
%% model parameters, as defined in the paper
M = 100; % data dimension
UNIFORM = 1; % 1 = model has uniform residual variances; 0 = models has arbitrary residual variances
OBJECTIVE = 1; % 0 = independent objective; 1 = aligned objective
scan_N = [ .25 .5 1 2 ] * M; %% the sizes of datasets
default_lambda = [2.8:0.2:3.4] * M; % the corresponding lambda to use
default_K = [ 0.02:0.01:0.17 ] * M; % the corresponding K to use
N_N = length( scan_N ); % number of scan points
N_K = length(default_K);
N_lambda = length(default_lambda);
TRIAL = 100; % number of simulation trials
mu_f = -1; % factor mean
sigma_f = 2; % factor standard deviation
sigma_p = 4; % object vector standard deviation
sigma_r = 0.6;
train_ratio = 0.7; % ratio of training samples in the dataset

%% set random seed for data generation; can be safely ignored
rand_seed = randi(10000);
fprintf('seed: %d\n', rand_seed);
rand_stream =  RandStream('mcg16807', 'Seed', rand_seed) ;
RandStream.setGlobalStream( rand_stream );

%% objective value record keeper
UTM_obj = zeros(TRIAL, N_N,N_K);
URM_obj = zeros(TRIAL, N_N, N_lambda);
oracle_obj = zeros(TRIAL, N_N);


%% begin of simulation
for trial = 1 : TRIAL
    [ X, ~, c ] = generate_data( M, scan_N(N_N), UNIFORM, OBJECTIVE, mu_f, sigma_f, sigma_p, sigma_r ); % X=data set; Sigma_s = true covariance matrix
            
    %% scan over different data sizes
    for index_N = 1 : N_N
        N = scan_N(index_N);
        N_train = round(N * train_ratio);
        N_test = N - N_train;
        %% compute sample covaraince matrix
        Sigma_SAM = zeros(M,M);
        for n = 1 : N_train
            Sigma_SAM = Sigma_SAM + X(:,n) * X(:,n)';
        end
        Sigma_SAM = Sigma_SAM / N_train;
                     
        Sigma_s = zeros(M,M);
        for n = N_train+1 : N
            Sigma_s = Sigma_s + X(:,n) * X(:,n)';
        end
        Sigma_s = Sigma_s / N_test;
        
        if UNIFORM == 1
            
            for index_K = 1:N_K
                train_K = default_K(index_K);
              %% URM
                [ Sigma_URM, F_URM, R_URM ] = compute_URM( Sigma_SAM, train_K );
                U_URM = 0.5 * (Sigma_URM \ c);
                URM_obj( trial, index_N, index_K ) = c' * U_URM - U_URM' * Sigma_s * U_URM;
            end
%             for index_lambda = 1:N_lambda
%                 train_lambda = default_lambda(index_lambda);
%               %% UTM
%                 [ Sigma_UTM, F_UTM, R_UTM ] = compute_UTM( Sigma_SAM, train_lambda, N_train );
%                 U_UTM = 0.5 * (Sigma_UTM \ c);
%                 UTM_obj( trial, index_N, index_lambda ) = c' * U_UTM - U_UTM' * Sigma_s * U_UTM;
%             end
        end     
    end    
end

% %% plot the results
% hd1 = figure('Position',[400 400 400 300]);
% color = {'bx','rx','gx','yx', 'kx'};
% if UNIFORM == 1
% %     for index_K = 1:N_K
% %         errorbar(scan_N, mean(URM_obj(:,:,index_K)), std(URM_obj(:,:,index_K))/sqrt(TRIAL), char(color(index_K)), 'MarkerSize', 4, 'DisplayName', num2str(default_K(index_K))); hold on;
% %     end
% %     for index_lambda = 1:N_lambda
% %     	errorbar(scan_N, mean(UTM_obj(:,:,index_lambda)), std(UTM_obj(:,:,index_lambda))/sqrt(TRIAL), char(color(index_lambda)), 'MarkerSize', 4, 'DisplayName', num2str(default_lambda(index_lambda))); hold on;
% %     end
% end
% xlabel('N');
% ylabel('Average performance');
% legend('Location', 'southeast');

for n = 1:N_N
    [val, m] = max(mean(URM_obj(:,n,:)));
    [m, default_K(m)]
end
toc