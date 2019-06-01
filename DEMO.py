# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import random
from algorithm import URM, UTM, PEO
from generate_data import generate_data
import matplotlib.pyplot as plt

M = 70 # data dimension
UNIFORM = True # True = model has uniform residual variances; False = models has arbitrary residual variances
OBJECTIVE = False # False = independent objective; True = aligned objective
scan_N = (np.array([0.25, 0.5, 1, 2]) * M).astype(np.int) # the sizes of datasets
default_K = np.array([5, 9, 15, 20]) # the corresponding K to use
default_lambda = np.array([2.6, 1.4, 1, .9]) * M # the corresponding lambda to use
default_gamma = np.array([8, 6, 4, 2]) * M
TRIAL = 100 # number of simulation trials
mu_f = -1
sigma_f = 2 # magnitude factor loadings
sigma_p = 4
sigma_r = 0.6 # magnitude of variation among residual variances

## log likelihood record keeper
UTM_llh = np.zeros((TRIAL, len(scan_N)))
URM_llh = np.zeros((TRIAL, len(scan_N)))
PEO_llh = np.zeros((TRIAL, len(scan_N)))
wrong_llh = np.zeros((TRIAL, len(scan_N)))

UTM_obj = np.zeros((TRIAL, len(scan_N)))
URM_obj = np.zeros((TRIAL, len(scan_N)))
PEO_obj = np.zeros((TRIAL, len(scan_N)))
Oracle_obj = np.zeros((TRIAL, len(scan_N)))

#UTM_llh_train = np.zeros((TRIAL, len(scan_N)))
#URM_llh_train = np.zeros((TRIAL, len(scan_N)))
#PEO_llh_train = np.zeros((TRIAL, len(scan_N)))
#
#UTM_obj_train = np.zeros((TRIAL, len(scan_N)))
#URM_obj_train = np.zeros((TRIAL, len(scan_N)))
#PEO_obj_train = np.zeros((TRIAL, len(scan_N)))

## set random seed for data generation; can be safely ignored
rand_seed = random.randint(0, 10000);
print("Seed: %d" % rand_seed)
np.random.seed(rand_seed)

# begin of simulation
for trial in tqdm.tqdm(range(TRIAL)):
    X, Sigma_s, c = generate_data(M, scan_N[-1], UNIFORM, OBJECTIVE, mu_f, sigma_f, sigma_p, sigma_r) # X=data set; Sigma_s = true covariance matrix
            
    for index_N, (N, train_K, train_lambda, train_gamma) in enumerate(zip(scan_N, default_K, default_lambda, default_gamma)):
        # compute sample covaraince matrix
        Sigma_SAM = np.zeros((M,M))
        for n in range(N):
            Sigma_SAM += np.outer(X[:,n], X[:,n])
        Sigma_SAM /= N
                     
        if UNIFORM:
            # URM
            Sigma_URM = URM.compute(Sigma_SAM, train_K)
            U_URM = 0.5 * np.linalg.lstsq(Sigma_URM, c, rcond=None)[0]
            URM_llh[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_URM)) + np.trace(np.linalg.lstsq(Sigma_URM, Sigma_s, rcond=None)[0]))
            URM_obj[trial, index_N] = c.dot(U_URM) - U_URM.dot(Sigma_s).dot(U_URM)
#            URM_llh_train[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_URM)) + np.trace(np.linalg.lstsq(Sigma_URM, Sigma_SAM, rcond=None)[0]))
#            URM_obj_train[trial, index_N] = c.dot(U_URM) - U_URM.dot(Sigma_SAM).dot(U_URM)
            
            # UTM
            Sigma_UTM = UTM.compute(Sigma_SAM, train_lambda, N)
            U_UTM = 0.5 * np.linalg.lstsq(Sigma_UTM, c, rcond=None)[0]
            UTM_llh[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_UTM)) + np.trace(np.linalg.lstsq(Sigma_UTM, Sigma_s, rcond=None)[0]))
            UTM_obj[trial, index_N] = c.dot(U_UTM) - U_UTM.dot(Sigma_s).dot(U_UTM)
#            UTM_llh_train[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_UTM)) + np.trace(np.linalg.lstsq(Sigma_UTM, Sigma_SAM, rcond=None)[0]))
#            UTM_obj_train[trial, index_N] = c.dot(U_UTM) - U_UTM.dot(Sigma_SAM).dot(U_UTM)

            # PEO
            Sigma_PEO = PEO.compute(Sigma_SAM, c, train_lambda, train_gamma, N)
            U_PEO = 0.5 * np.linalg.lstsq(Sigma_PEO, c, rcond=None)[0]
            PEO_llh[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_PEO)) + np.trace(np.linalg.lstsq(Sigma_PEO, Sigma_s, rcond=None)[0]))
            PEO_obj[trial, index_N] = c.dot(U_PEO) - U_PEO.dot(Sigma_s).dot(U_PEO)
#            PEO_llh_train[trial, index_N] = -0.5 * (M * np.log(2*np.pi) + np.log(np.linalg.det(Sigma_PEO)) + np.trace(np.linalg.lstsq(Sigma_PEO, Sigma_SAM, rcond=None)[0]))
#            PEO_obj_train[trial, index_N] = c.dot(U_PEO) - U_PEO.dot(Sigma_SAM).dot(U_PEO)
            
            U_oracle = 0.5 * np.linalg.lstsq(Sigma_s, c, rcond=None)[0]
            Oracle_obj[trial, index_N] = c.dot(U_oracle) - U_oracle.dot(Sigma_s).dot(U_oracle)
       
        
# plot the results
log_scan_N = np.log(scan_N/M)
plt.figure(figsize = (10,4))
plt.subplot(121)
if UNIFORM:
    plt.errorbar(log_scan_N, URM_llh.mean(axis=0), URM_llh.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='r', ms=4, label='URM')
    plt.errorbar(log_scan_N, UTM_llh.mean(axis=0), UTM_llh.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='b', ms=4, label='UTM')
    plt.errorbar(log_scan_N, PEO_llh.mean(axis=0), PEO_llh.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='g', ms=4, label='PEO')
    
plt.xlabel('log(N/M)');
plt.ylabel('log likelihood');
plt.legend()


plt.subplot(122)
if UNIFORM:
    plt.errorbar(log_scan_N, URM_obj.mean(axis=0), URM_obj.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='r', ms=4, label='URM')
    plt.errorbar(log_scan_N, UTM_obj.mean(axis=0), UTM_obj.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='b', ms=4, label='UTM')
    plt.errorbar(log_scan_N, PEO_obj.mean(axis=0), PEO_obj.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='g', ms=4, label='PEO')
    plt.errorbar(log_scan_N, Oracle_obj.mean(axis=0), Oracle_obj.std(axis=0, ddof=1)/np.sqrt(TRIAL), c='k', ms=4, label='Oracle')
    
plt.xlabel('log(N/M)');
plt.ylabel('Score');
plt.legend()
plt.show()