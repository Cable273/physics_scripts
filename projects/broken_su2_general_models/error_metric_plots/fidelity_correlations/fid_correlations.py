#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

x = dict()
x[0] = np.load("./pxp,z2,su2_errors,18.npy")
x[1] = np.load("./pxp,z3,su2_errors,18.npy")
x[2] = np.load("./pxp,z4,su2_errors,16.npy")
x[3] = np.load("./pxp,z2,su2_errors,12.npy")
x[4] = np.load("./pcp,z2,su3_errors,14.npy")
x[5] = np.load("./pxyp,su2_errors,20.npy")
x[6] = np.load("./ising,su2_errors,14.npy")

print(x[0])
f_pxpSpinHalfZ2 = x[0][:,0]
sVar_pxpSpinHalfZ2 = x[0][:,1]/(18)
maxVar_pxpSpinHalfZ2 = x[0][:,2]/(18)

print(x[1])
f_pxpSpinHalfZ3 = x[1][:,0]
sVar_pxpSpinHalfZ3 = x[1][:,1]/(2*18/3)
maxVar_pxpSpinHalfZ3 = x[1][:,2]/(2*18/3)

print(x[2])
f_pxpSpinHalfZ4 = x[2][:,0]
sVar_pxpSpinHalfZ4 = x[2][:,1]/(2*16/4)
maxVar_pxpSpinHalfZ4 = x[2][:,2]/(2*16/4)

print(x[3])
f_pxpSpinOneZ2 = x[3][:,0]
sVar_pxpSpinOneZ2 = x[3][:,1]/(2*12)
maxVar_pxpSpinOneZ2 = x[3][:,2]/(2*13)

f_pcpZ2 = x[4][:,0]
sVar_pcpZ2 = x[4][:,1]/(2*14)

plt.scatter(sVar_pxpSpinHalfZ2,1-f_pxpSpinHalfZ2,label="PXP Spin half Z2",alpha=0.6,s=100)
plt.scatter(sVar_pxpSpinHalfZ3,1-f_pxpSpinHalfZ3,marker="D",label="PXP Spin half Z3",alpha=0.6,s=100)
plt.scatter(sVar_pxpSpinHalfZ4,1-f_pxpSpinHalfZ4,marker="s",label="PXP Spin half Z4",alpha=0.6,s=100)
plt.scatter(sVar_pxpSpinOneZ2,1-f_pxpSpinOneZ2,marker="x",label="PXP Spin one, Z2",alpha=0.6,s=100)
plt.legend()
plt.xlabel(r"$\sigma/D_{rep}$")
plt.ylabel(r"$f_0$")
plt.grid(True)
plt.show()

plt.scatter(maxVar_pxpSpinHalfZ2,1-f_pxpSpinHalfZ2,label="PXP Spin half Z2",alpha=0.6,s=100)
plt.scatter(maxVar_pxpSpinHalfZ3,1-f_pxpSpinHalfZ3,marker="D",label="PXP Spin half Z3",alpha=0.6,s=100)
plt.scatter(maxVar_pxpSpinHalfZ4,1-f_pxpSpinHalfZ4,marker="s",label="PXP Spin half Z4",alpha=0.6,s=100)
plt.scatter(maxVar_pxpSpinOneZ2,1-f_pxpSpinOneZ2,marker="x",label="PXP Spin one, Z2",alpha=0.6,s=100)
plt.legend()
plt.xlabel(r"$\max(var(H^z)_n)/D_{rep}$")
plt.ylabel(r"$f_0$")
plt.grid(True)
plt.show()
