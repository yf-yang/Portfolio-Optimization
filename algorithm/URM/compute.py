# -*- coding: utf-8 -*-

import numpy as np
from ..common import ordered_eig

def compute(Sigma_SAM, K):
    M = Sigma_SAM.shape[0]
    eigval, eigvec = ordered_eig(Sigma_SAM)
    sigma_URM = eigval[K:].mean()
    F_URM = eigvec[:,:K].dot(np.diag(eigval[:K]- sigma_URM)).dot(eigvec[:,:K].T)
    R_URM = sigma_URM * np.eye(M)
    Sigma_URM = F_URM + R_URM
    return Sigma_URM