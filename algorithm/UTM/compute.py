# -*- coding: utf-8 -*-

import numpy as np
from ..common import trans

def compute(Sigma_SAM, lamb, N):
    lambda_p = 2 * lamb / N
    Sigma_UTM, _, _, _ = trans(Sigma_SAM, lambda_p)
    return Sigma_UTM