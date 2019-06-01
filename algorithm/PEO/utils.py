# -*- coding: utf-8 -*-

import numpy as np
from ..common import trans

__all__ = ['is_pos_def', 'f', 'df', 'CD_op', 'get_v_G']

def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)

def f(c, Sigma_SAM, gamma, lamb, G, v):
    M = Sigma_SAM.shape[0]
    Sigma_inv = v * np.eye(M) - G
    return (gamma * (c.dot(Sigma_inv).dot(c) - 0.5 * c.dot(Sigma_inv).dot(Sigma_SAM).dot(Sigma_inv).dot(c))
            + np.linalg.slogdet(Sigma_inv)[1] - np.trace(Sigma_inv.dot(Sigma_SAM)) 
            - lamb * np.trace(G))
    
def df(c, Sigma_SAM, gamma, lamb, G, v):
    M = Sigma_SAM.shape[0]
    Sigma_inv = v * np.eye(M) - G
    Sigma = np.linalg.inv(Sigma_inv)
    C_op_D = CD_op(c, Sigma_SAM, G, v)
    dG = gamma * C_op_D + Sigma_SAM - Sigma - lamb * np.eye(M)
    dv = np.trace(Sigma - Sigma_SAM - gamma * C_op_D)
    return dv, dG

def CD_op(c, Sigma_SAM, G, v):
    M = Sigma_SAM.shape[0]
    C = np.outer(c,c)
    Sigma_inv = v * np.eye(M) - G
    D = np.dot(Sigma_inv, Sigma_SAM) - np.eye(M)
    CD = C.dot(D)
    return 0.5 * (CD + CD.T)
    
def get_v_G(Sigma, lamb):
    tmp, eigvec, eigval, v_inv = trans(Sigma, lamb)
    v = 1 / v_inv
    eigval = v - 1 / (eigval + v_inv)
    G = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    return v, G