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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT,inversion
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
N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.model = np.array([[0,1,1,1,0]])
H.model_coef = np.array([1])
H.gen()

Hp = Hamiltonian(pxp,pxp_syms)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,1,2,1,0],[0,1,2,1,0],[0,2,1,2,0],[0,1,2,1,0]])
Hp.model_coef = np.array([1,1,1,1])
Hp.uc_size = np.array([4,4,4,4])
Hp.uc_pos = np.array([2,3,0,1])
Hp.gen()

from Calculations import gen_fsa_basis
z=zm_state(4,1,pxp)
fsa_basis = gen_fsa_basis(Hp.sector.matrix(),z.prod_basis(),int(2*pxp.N/4))
print(np.shape(fsa_basis))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e,u = np.linalg.eigh(H_fsa)
fsa_energy = e
fsa_overlap = np.log10(np.abs(u[0,:])**2)
plt.scatter(fsa_energy,fsa_overlap)
plt.show()

H.sector.find_eig()
ent = entropy(pxp)
ent_vals = np.zeros(pxp.dim)
for n in range(0,np.size(pxp.basis,axis=0)):
    ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
plt.scatter(H.sector.eigvalues(),ent_vals)
plt.show()
    

# k=pxp_syms.find_k_ref(z.ref)
# exact_energy = []
# exact_overlap = []
# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])
    # exact_energy = np.append(exact_energy,H.sector.eigvalues(k[n]))
    # exact_overlap = np.append(exact_overlap,eig_overlap(z,H,k[n]).eval())

# plt.scatter(exact_energy,exact_overlap)
# plt.ylim(bottom=-10)
# plt.show()
# fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
# plt.show()

# np.save("pxxxp,0th_order,e,"+str(pxp.N),exact_energy)
# np.save("pxxxp,0th_order,LW_overlap,"+str(pxp.N),exact_overlap)

np.save("pxxxp,LW_fsa,0th_order,e,"+str(pxp.N),fsa_energy)
np.save("pxxxp,LW_fsa,0th_order,LW_overlap,"+str(pxp.N),fsa_overlap)
