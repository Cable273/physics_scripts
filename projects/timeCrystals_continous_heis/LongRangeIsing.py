#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

#init system
N=16
pxp = unlocking_System([0,1],"periodic",2,N,)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

J = -1
B = 0.27
alpha = 2.3
#create Hamiltonian
model =[]
model_coef = []
for n in range(2,N+1):
    temp = -np.ones(n)
    temp[0] = 1
    temp[np.size(temp,axis=0)-1] = 1
    d = np.size(temp)-1
    if d > int(N/2):
        d = N - d
    model_coef = np.append(model_coef,J/np.power(d,alpha))
    model.append(temp)
Hz = Hamiltonian(pxp,pxp_syms)
Hz.site_ops[1] = np.array([[-1,0],[0,1]])
Hz.model = model
Hz.model_coef = model_coef

Hx = Hamiltonian(pxp,pxp_syms)
Hx.site_ops[1] = np.array([[0,1],[1,0]])
Hx.model = np.array([[1]])
Hx.model_coef = np.array([-B])

k=[0]
z=ref_state(np.max(pxp.basis_refs),pxp)
Hz.gen(k)
Hx.gen(k)
H = H_operations.add(Hz,Hx,np.array([1,1]))
H.sector.find_eig(k)

overlap = eig_overlap(z,H,k).eval()
plt.scatter(H.sector.eigvalues(k),overlap)
plt.title(r"$H = -\sum_{i<j}^L \frac{J}{r_{ij}^\alpha} \sigma_i^z \sigma_{i+1}^z - B \sum_i \sigma_i^x$"+"\n"+r"$J=1$, $B=0.27$, $\alpha=2.3$, $k=0$, $N=$"+str(pxp.N))
plt.ylabel(r"$\log(\vert \langle 1111... \vert E \rangle \vert^2)$")
plt.xlabel(r"$E$")

plt.show()

