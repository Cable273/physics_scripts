#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

N=12
exact_energy = np.load("./pcp,2ndOrder,energy,"+str(N)+".npy")
exact_overlap = np.load("./pcp,2ndOrder,overlap,"+str(N)+".npy")
su3_energy = np.load("./pcp,2ndOrder,fsa_energy,"+str(N)+".npy")
# su3_overlap = np.load("./pcp,2ndOrder,fsa_exact_overlap,"+str(N)+".npy")
su3_overlap = np.load("./pcp,2ndOrder,fsa_overlap,"+str(N)+".npy")
plt.scatter(exact_energy,exact_overlap)
plt.scatter(su3_energy,su3_overlap,marker="x",s=100,color="red")
plt.show()

t=np.arange(0,20,0.01)
f=np.load("./pcp,2ndOrder,fidelity,16.npy")
plt.plot(t,f)
plt.show()

