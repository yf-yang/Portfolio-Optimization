clear all;

%% model parameters, as defined in the paper
M = 200; % data dimension
K = 0.05 * M; % number of factors
UNIFORM = 1; % 1 = model has uniform residual variances; 0 = models has arbitrary residual variances
scan_N = [ 0.25 0.5 1 2 ] * M; %% the sizes of datasets
default_lambda = [ 0.8 1.2 1.6 2 ] * M ; % the corresponding lambda to use
default_K = [ K-3 K-2 K-1 K ]; % the corresponding K to use
N_N = length( scan_N ); % number of scan points
TRIAL = 100; % number of simulation trials
sigma_f = 5; % magnitude factor loa  dings
sigma_r = 0.8; % magnitude of variation among residual variances

%% set random seed for data generation; can be safely ignored
rand_seed = randi(10000);
fprintf('seed: %d\n', rand_seed);
rand_stream =  RandStream('mcg16807', 'Seed', rand_seed) ;
RandStream.setGlobalStream( rand_stream );

%% log likelihood record keeper
UTM_llh = zeros(TRIAL, N_N);
URM_llh = zeros(TRIAL, N_N);
MRH_llh = zeros(TRIAL, N_N);
EM_llh = zeros(TRIAL, N_N);
TM_llh = zeros(TRIAL, N_N);
STM_llh = zeros(TRIAL, N_N);


%% begin of simulation
for trial = 1 : TRIAL
    [ X, Sigma_s ] = generate_data( M, K, scan_N(N_N), UNIFORM, sigma_f, sigma_r ); % X=data set; Sigma_s = true covariance matrix
            
    %% scan over different data sizes
    for index_N = 1 : N_N
        N = scan_N(index_N);
        train_K = default_K( index_N );
        train_lambda = default_lambda( index_N );
        %% compute sample covaraince matrix
        Sigma_SAM = zeros(M,M);
        for n = 1 : N
            Sigma_SAM = Sigma_SAM + X(:,n) * X(:,n)';
        end
        Sigma_SAM = Sigma_SAM / N;
                     
        if UNIFORM == 1
            %% URM
            [ Sigma_URM, F_URM, R_URM ] = compute_URM( Sigma_SAM, train_K );
            URM_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_URM) + trace(Sigma_URM\Sigma_s) );
           
            %% UTM
            [ Sigma_UTM, F_UTM, R_UTM ] = compute_UTM( Sigma_SAM, train_lambda, N );
            UTM_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_UTM) + trace(Sigma_UTM\Sigma_s) );
            
        else   
            %% MRH
            [ Sigma_MRH, F_MRH, R_MRH ] = compute_MRH( Sigma_SAM, train_K );
            MRH_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_MRH) + trace(Sigma_MRH\Sigma_s) );
             
            %% EM
            [ Sigma_EM, F_EM, R_EM ] = compute_EM( Sigma_SAM, train_K );
            EM_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_EM) + trace(Sigma_EM\Sigma_s) );
            
            %% TM
            [ Sigma_TM, F_TM, R_TM ] = compute_TM( Sigma_SAM, train_lambda, N );
            TM_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_TM) + trace(Sigma_TM\Sigma_s) );
            
            %% STM
            [ Sigma_STM, F_STM, R_STM ] = compute_STM( Sigma_SAM, train_lambda, N );
            STM_llh( trial, index_N ) = -0.5 * ( M * log(2*pi) + logdet(Sigma_STM) + trace(Sigma_STM\Sigma_s) );
           
        end       
    end    
end

%% plot the results
log_scan_N = log( scan_N/M);
hd1 = figure('Position',[400 400 400 300]);
legend_str = [];
if UNIFORM == 1
    errorbar(log_scan_N, mean(URM_llh), std(URM_llh)/sqrt(TRIAL), 'r', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'URM'];
    errorbar(log_scan_N, mean(UTM_llh), std(UTM_llh)/sqrt(TRIAL), 'b', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'UTM'];
else
    errorbar(log_scan_N, mean(EM_llh), std(EM_llh)/sqrt(TRIAL), 'r*-', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'EM '];
    errorbar(log_scan_N, mean(MRH_llh), std(MRH_llh)/sqrt(TRIAL), 'gs-', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'MRH'];
    errorbar(log_scan_N, mean(TM_llh), std(TM_llh)/sqrt(TRIAL), 'ko-', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'TM '];
    errorbar(log_scan_N, mean(STM_llh), std(STM_llh)/sqrt(TRIAL), 'b-', 'MarkerSize', 4 ); hold on; legend_str = [legend_str ; 'STM'];
end
xlabel('log(N/M)');
ylabel('log likelihood');
legend(legend_str);
