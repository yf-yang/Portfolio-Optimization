# -*- coding: utf-8 -*-

import numpy as np
from .utils import *

stop_threshold = 1e-5

def compute(Sigma_SAM, c, lamb, gamma, N):
    M = Sigma_SAM.shape[0]
    lamb = 2 * lamb / N
    gamma = 2 * gamma / N
    v, G = get_v_G(Sigma_SAM, lamb)
    while True:
        C_op_D = CD_op(c, Sigma_SAM, G, v)
        v_new, G_new = get_v_G(Sigma_SAM + gamma * C_op_D, lamb)
        
        Sigma_inv = v * np.eye(M) - G
        Sigma = np.linalg.inv(Sigma_inv)
        Sigma_new = np.linalg.inv(v_new * np.eye(M) - G_new)
        
        if np.linalg.norm(Sigma - Sigma_new) < stop_threshold * np.linalg.norm(Sigma):
            break
        
        delta_v, delta_G = v_new - v, G_new - G
        
        posdef = is_pos_def(Sigma_new)
        
        objective = f(c, Sigma_SAM, gamma, lamb, G, v)
        g = gamma * c.dot(Sigma_inv).dot(c) - 0.5 * c.dot(Sigma_inv).dot(Sigma_SAM).dot(Sigma_inv).dot(c)
        llh = np.linalg.slogdet(Sigma_inv)[1] - np.trace(Sigma_inv.dot(Sigma_SAM)) 
        tr = - lamb * np.trace(G)
        if posdef:
            alpha = 1
            while True:
                dv, dG = df(c, Sigma_SAM, gamma, lamb, G, v)
                if (
                    f(c, Sigma_SAM, gamma, lamb, G+alpha*delta_G, v+alpha*delta_v) <
                    objective +
                    0.1 * alpha * ((dG * delta_G).sum() + dv * delta_v)
                    ):
                    alpha /= 2
                else:
                    break
                if alpha < 1e-6:
                    alpha = 0
                    break
            if alpha:
                v, G = v + alpha * delta_v, G + alpha * delta_G
        
        if not (posdef and alpha):
            alpha = 1
            while True:
                dv, dG = df(c, Sigma_SAM, gamma, lamb, G, v)
                v_new, G_new = v + alpha * dv, G + alpha * dG
                eigenValues, eigenVectors = np.linalg.eigh(G_new)
                eigenValues = np.maximum(eigenValues, 0)
                G_new = eigenVectors.dot(np.diag(eigenValues)).dot(eigenVectors.T)
                if f(c, Sigma_SAM, gamma, lamb, G_new, v_new) > objective:
                    break
                alpha /= 2
            v, G = v_new, G_new
        
    Sigma_PEO = Sigma

    return Sigma_PEO