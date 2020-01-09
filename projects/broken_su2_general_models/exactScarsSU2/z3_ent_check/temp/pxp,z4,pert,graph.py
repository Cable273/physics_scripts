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
from copy import deepcopy

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
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
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=4)])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,2,0],[0,1,0],[0,1,0],[0,1,0]])
Hp[0].model_coef = np.array([1,1,1,1])
Hp[0].uc_size = np.array([4,4,4,4])
Hp[0].uc_pos = np.array([3,0,1,2])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,1,2,1,0]])
Hp[1].model_coef = np.array([1])
Hp[1].uc_size = np.array([4])
Hp[1].uc_pos = np.array([0])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[2].model_coef = np.array([1,1])
Hp[2].uc_size = np.array([4,4])
Hp[2].uc_pos = np.array([3,1])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([4,4,4,4])
Hp[3].uc_pos = np.array([0,0,1,1])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Hp[4].model_coef = np.array([1,1,1,1])
Hp[4].uc_size = np.array([4,4,4,4])
Hp[4].uc_pos = np.array([2,3,3,2])

z=zm_state(4,1,pxp)
print("Z4 label:")
print(pxp.keys[z.ref])
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,len(Hp)):
    Hp[n].gen()

coef = np.zeros(4)
coef[1] = -1
# coef = np.load("../../../../pxp,2nd_order_perts/z4_first_order/data/pxp,z4,pert_coef.npy")
# coef = np.load("../../../../pxp,2nd_order_perts/z4_2nd_order/pxp,z4,2nd_order_pert_coef,12.npy")
# coef = np.load("./pxp,z4,2nd_order_pert_coef,12.npy")
# coef = np.array([0.18243653,-0.10390499,0.0544521])
# coef = np.loadtxt("../../../../pxp,2nd_order_perts/z3/data/pxp,z3,2nd_order_pert,coef,18")
print(coef)

Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hm = Hp_total.herm_conj()

H = H_operations.add(Hp_total,Hm,np.array([1,1]))
H.sector.find_eig()
for n in range(0,np.size(H.sector.matrix(),axis=0)):
    for m in range(0,np.size(H.sector.matrix(),axis=0)):
        if np.abs(H.sector.matrix()[n,m]) <1e-5:
            H.sector.matrix()[n,m] = 0
    
eig_overlap(z,H).plot(tol=-10)
plt.show()
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()

from Calculations import gen_krylov_basis
kbasis = gen_krylov_basis(H.sector.matrix(),pxp.dim,z.prod_basis())
print("krylov dim |z4>:")
print(np.shape(kbasis))

from Calculations import plot_adjacency_graph
plot_adjacency_graph(np.abs(H.sector.matrix()),labels=None,largest_comp=False)
plt.show()
