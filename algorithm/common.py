# -*- coding: utf-8 -*-

import numpy as np

def ordered_eig(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]   
    return eigenValues[idx], eigenVectors[:,idx]

def trans(Sigma_SAM, lamb):
    M = Sigma_SAM.shape[0]
    eigval, eigvec = ordered_eig(Sigma_SAM)
    for i in range(M-1):
        beta = ((i+1)*lamb + sum(eigval[i+1:M]) ) / (M-i-1)
        if beta > eigval[i] - lamb:
            break
    K = i
    beta = (K*lamb + sum(eigval[K:M])) / (M-K)
    # original source code is wrong
    eigval_new = eigval[:K]-lamb-beta
    F_trans = eigvec[:,:K].dot(np.diag(eigval_new)).dot(eigvec[:,:K].T)
    R_trans = beta * np.eye(M)
    Sigma = F_trans + R_trans
    return Sigma, eigvec[:,:K], eigval_new, beta