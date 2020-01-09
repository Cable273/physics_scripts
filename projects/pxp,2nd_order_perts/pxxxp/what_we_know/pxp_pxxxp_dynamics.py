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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
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

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
# H.model = np.array([[0,1,0],[0,1,1,1,0]])
H.model = np.array([[0,1,0],[0,0,1,0],[0,1,0,0]])
# H.model_coef = np.array([1,0.122959959])
a=0.108
H.model_coef = np.array([1,a,a])

z=zm_state(2,1,pxp)

k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    eig_overlap(z,H,k[n]).plot()
plt.title(r"$H=PXP + \lambda PXXXP$, N="+str(pxp.N))
plt.show()

t=np.arange(0,20,0.001)
f0 = fidelity(z,H,"use sym").eval(t,z)
for n in range(0,np.size(f0,axis=0)):
    if f0[n] < 0.1:
        cut = n
        break
max_index = np.argmax(f0[cut:])
print(t[cut:][max_index])

# plt.plot(t,f0)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# plt.title(r"$H=PXP + \lambda PXXXP$, N="+str(pxp.N))
# plt.show()
