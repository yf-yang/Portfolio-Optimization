# -*- coding: utf-8 -*-

import numpy as np

def generate_data(M, N, uniform, objective, mu_f, sigma_f, sigma_p, sigma_r):
    ## generate basis ( no effect if isotropic noise is assumed )
    Psi = np.random.randn(M, M);
    for d1 in range(M):
        for d2 in range(d1-1):
            Psi[:,d1] -= Psi[:,d2].dot(Psi[:,d1]) * Psi[:,d2]
        Psi[:,d1] /= np.linalg.norm(Psi[:,d1])

    ## generate factor coefficients
    f = np.random.randn(M) * sigma_f + mu_f
    f = np.exp(f)
    f[::-1].sort()

    ## generate residuals (uniform / inverse-gamma)
    if uniform:
        R_s = np.ones(M)
    else:
        ## inverse gamma
        R_s = np.exp(np.random.randn(M) * sigma_r)
    R_sqrt = np.sqrt(R_s)
    R_s = np.diag(R_s)

    ## generate c
    c = np.random.randn(M)
    if objective:
        p = np.random.randn(20) * sigma_p
        for d in range(20):
            c += p[d]*Psi[:,d]
    c = c / np.linalg.norm(c)
    
    ## generate Sigma_s
    Fhalf = np.zeros((M, M))
    for d in range(M):
        Fhalf[:,d] = f[d]*Psi[:,d]
    F_s = Fhalf.dot(Fhalf.T)
    Sigma_s = F_s + R_s

    ## generate data
    X = np.zeros((M, N))
    for n in range(N):
        zn = np.random.randn(M)
        wn = np.random.randn(M)
        X[:,n] = Fhalf.dot(zn) + wn*R_sqrt
    return X, Sigma_s, c