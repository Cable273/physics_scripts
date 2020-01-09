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
H.model = np.array([[0,1,1,1,0]])
H.model_coef = np.array([1])

z=zm_state(4,1,pxp)
z1=zm_state(4,1,pxp,1)
z2=zm_state(4,1,pxp,2)
z3=zm_state(4,1,pxp,3)

k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    eig_overlap(z,H,k[n]).plot()
plt.show()

t=np.arange(0,20,0.01)
f0 = fidelity(z,H,"use sym").eval(t,z)
f1 = fidelity(z,H,"use sym").eval(t,z1)
f2 = fidelity(z,H,"use sym").eval(t,z2)
f3 = fidelity(z,H,"use sym").eval(t,z3)

plt.plot(t,f0,label=r"$\vert 10001000...\rangle$")
plt.plot(t,f1,label=r"$\vert 01000100...\rangle$")
plt.plot(t,f2,label=r"$\vert 00100010...\rangle$")
plt.plot(t,f3,label=r"$\vert 00010001...\rangle$")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \phi \vert e^{-iHt} \vert \psi(0) \rangle \vert^2$")
plt.title(r"$H=PXXXP$, N="+str(pxp.N))
plt.show()
