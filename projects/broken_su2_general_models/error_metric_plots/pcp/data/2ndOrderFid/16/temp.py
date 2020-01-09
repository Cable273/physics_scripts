#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

f=np.load("./pcp,2ndOrder,fidelity,16.npy")
for n in range(0,np.size(f,axis=0)):
    if f[n]<0.1:
        cut = n
        break
print(np.max(f[cut:]))
    

