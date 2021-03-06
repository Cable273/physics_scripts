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

from Hamiltonian_Classes import *
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import *
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

N = 21
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=3)])
z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
all_k = k

def sym_key(k):
    return bin_to_int_base_m(k,pxp.N+1)
U = dict()
c=0
for n in range(0,np.size(all_k,axis=0)):
    U[sym_key(all_k[n])] = pxp_syms.basis_transformation(all_k[n])
    c = c + np.shape(U[sym_key(all_k[n])])[1]

# dynamics + fidelity
V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,0,0]])
V1_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=1)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[1].model = np.array([[0,0,1,0]])
V1_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[2].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[3].gen(k_vec=k[n],uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,0,1,0]])
V2_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V2_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,0,0]])
V2_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V2_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,1,1,1,0]])
V3_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V3_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,1,1,0]])
V3_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V3_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
for n in range(0,np.size(k,axis=0)):
    H0.gen(k[n])
coef = np.array([0.18243653,-0.10390499,0.054452])
H = H_operations.add(H0,V1,np.array([1,coef[0]]))
H = H_operations.add(H,V2,np.array([1,coef[1]]))
H = H_operations.add(H,V3,np.array([1,coef[2]]))
for n in range(0,np.size(k,axis=0)):
    H.sector.find_eig(k[n])

# rotate to full basis and direct sum sym eigenstates
eigvectors_comp = dict()
for n in range(0,np.size(all_k,axis=0)):
    eigvectors_comp[sym_key(all_k[n])] = np.dot(U[sym_key(all_k[n])],H.sector.eigvectors(all_k[n]))

eigvectors_comp_comb = eigvectors_comp[sym_key(all_k[0])]
eigvalues_comb = H.sector.eigvalues(all_k[0])
for n in range(1,np.size(all_k,axis=0)):
    eigvectors_comp_comb = np.hstack((eigvectors_comp_comb,eigvectors_comp[sym_key(all_k[n])]))
    eigvalues_comb = np.append(eigvalues_comb,H.sector.eigvalues(all_k[n]))

# overlap = np.log10(np.abs(eigvectors_comp_comb[pxp.keys[z.ref],:])**2)
# plt.scatter(eigvalues_comb,overlap)
# plt.show()
# fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
# plt.show()
np.save("pxp,pert,e,"+str(pxp.N),eigvalues_comb)
np.save("pxp,pert,u,"+str(pxp.N),eigvectors_comp_comb)
