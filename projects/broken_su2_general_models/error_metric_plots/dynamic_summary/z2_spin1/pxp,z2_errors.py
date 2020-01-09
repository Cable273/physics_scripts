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
from Symmetry_Classes import *
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
pxp = unlocking_System([0],"periodic",3,N)
pxp.gen_basis()
pxp_syms= model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

s=1
m=np.arange(-s,s)
couplings = np.power(s*(s+1)-m*(m+1),0.5)
sp = np.diag(couplings,-1)
sm = np.diag(couplings,1)

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
# Hp[0] = Hamiltonian(pxp)
Hp[0].site_ops[1] = sp
Hp[0].site_ops[2] = sm
Hp[0].model = np.array([[0,1,0],[0,2,0]])
Hp[0].model_coef = np.array([1,1])
Hp[0].uc_size = np.array([2,2])
Hp[0].uc_pos = np.array([1,0])

Hp[1] = Hamiltonian(pxp,pxp_syms)
# Hp[1] = Hamiltonian(pxp)
# Hp[1].site_ops[1] = np.array([[0,0,0],[1,0,0],[0,0,0]])
# Hp[1].site_ops[2] = np.array([[0,1,0],[0,0,0],[0,0,0]])
# Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,2,0],[0,2,0,0]])
# Hp[1].model_coef = np.array([1,1,1,1])
# Hp[1].uc_size = np.array([2,2,2,2])
# Hp[1].uc_pos = np.array([0,1,1,0])

Hp[1].site_ops[1] = sp
Hp[1].site_ops[2] = sm
Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,2,0],[0,2,0,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2])
Hp[1].uc_pos = np.array([0,1,1,0])

z=zm_state(2,2,pxp,1)
k=pxp_syms.find_k_ref(z.ref)

for n in range(0,len(Hp)):
    Hp[n].gen()
    # for m in range(0,np.size(k,axis=0)):
        # Hp[n].gen(k[m])

coef = np.zeros(1)
# coef = np.load("../../../../pxp,2nd_order_perts/z2/data/all_terms/18/18_all_terms/pxp,z2,2nd_order_perts,fid_coef,18.npy")
# val = np.load("../../z2_spin1/pxp,spin1,z2,fid_coef,10.npy")
val = np.load("../../z2_spin1/pxp,spin1,z2,fid_coef,ppxp,10.npy")
coef[0] = val
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hm = Hp_total.herm_conj()
H = H_operations.add(Hp_total,Hm,np.array([1,1]))

# exact_energy = []
# exact_overlap = []
# for n in range(0,np.size(k,axis=0)):
    # H.sector.find_eig(k[n])
    # exact_energy = np.append(exact_energy,H.sector.eigvalues(k[n]))
    # exact_overlap = np.append(exact_overlap,eig_overlap(z,H,k[n]).eval())
# t=np.arange(0,20,0.01)
# f = fidelity(z,H,"use sym").eval(t,z)
# plt.scatter(exact_energy,exact_overlap)
# plt.show()
# plt.plot(t,f)
# plt.show()

fsa_dim = 2*pxp.N
from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),fsa_dim)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))

# # H.sector.find_eig()
# # exact_overlap = eig_overlap(z,H).eval()
# # exact_energy = H.sector.eigvalues()

e,u = np.linalg.eigh(H_fsa)
fsa_overlap = np.log10(np.abs(u[0,:])**2)
fsa_energy = e

# plt.scatter(exact_energy,exact_overlap)
# plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
# plt.show()

# t=np.arange(0,20,0.01)
# f=fidelity(z,H).eval(t,z)
# plt.plot(t,f)
# plt.show()

# np.save("pxp,0th_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,0th_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,0th_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,0th_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,0th_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pxp,1st_order,e,ppxp,"+str(pxp.N),exact_energy)
# np.save("pxp,1st_order,z2_overlap,ppxp,"+str(pxp.N),exact_overlap)
# np.save("pxp,1st_order,z2_fidelity,ppxp,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,1st_order,e,ppxp,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,1st_order,z2_overlap,ppxp,"+str(pxp.N),fsa_overlap)

# np.save("pxp,1st_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,1st_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,1st_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,1st_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,1st_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pxp,2nd_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,2nd_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,2nd_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,2nd_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,2nd_order,z2_overlap,"+str(pxp.N),fsa_overlap)
