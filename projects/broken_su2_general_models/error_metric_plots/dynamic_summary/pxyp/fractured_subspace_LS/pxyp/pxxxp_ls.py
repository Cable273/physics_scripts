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
N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp = pxp.U1_sector(4)
print("DIM = "+str(pxp.dim))
# pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[0,0]])
H.site_ops[2] = np.array([[0,0],[1,0]])
H.model = np.array([[0,1,2,0],[0,2,1,0]])
H.model_coef = np.array([1,1])
H.gen()

from Calculations import plot_adjacency_graph
plot_adjacency_graph(np.abs(H.sector.matrix()),labels=None,largest_comp=False)
plt.show()

Hp = Hamiltonian(pxp,pxp_syms)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,1,2,0],[0,2,1,0]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([2,2])
Hp.uc_pos = np.array([1,0])
Hp.gen()
Hm = Hp.herm_conj()
Hz = 1/2*com(Hp.sector.matrix(),Hm.sector.matrix())
e,u = np.linalg.eigh(Hz)
psi = u[:,0]

H.sector.find_eig()
psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),psi)
overlap = np.log10(np.abs(psi_energy)**2)
plt.scatter(H.sector.eigvalues(),overlap)
plt.ylim(bottom=-10)
plt.show()

    
# from Calculations import gen_krylov_basis
# k=[0,0]
# H.gen(k)
# krylovBasis = gen_krylov_basis(H.sector.matrix(k),pxp.dim,z.sym_basis(k,pxp_syms))
# H_krylov = np.dot(np.conj(np.transpose(krylovBasis)),np.dot(H.sector.matrix(k),krylovBasis))
# e,u = np.linalg.eigh(H_krylov)
# ls = level_stats(e)
# print(np.shape(H_krylov))
# print(ls.mean())

# krylovBasis = dict()
# for n in range(0,np.size(k,axis=0)):
    # krylovBasis[n] = gen_krylov_basis(H.sector.matrix(k[n]),pxp.dim,z.sym_basis(k[n],pxp_syms))

# H_krylov = dict()
# for n in range(0,np.size(k,axis=0)):
    # H_krylov[n] = np.dot(np.conj(np.transpose(krylovBasis[n])),np.dot(H.sector.matrix(k[n]),krylovBasis[n]))
# print(np.shape(H_krylov[0]))
# e,u = np.linalg.eigh(H_krylov[0])
# ls = level_stats(e)
# print(ls.mean())

# from Calculations import connected_comps
# comp = connected_comps(H)
# comp.find_connected_components()

# find sector with z4 state
# for n in range(0,len(comp.components)):
    # if z.ref in comp.components[n]:
        # largest_component = comp.components[n]
        # break
# np.save("pxxxp,lcc,"+str(pxp.N),largest_component)
    
# reset basis to subspace
# pxp.basis_refs = np.sort(largest_component)
# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# pxp.keys = dict()
# pxp.dim = np.size(pxp.basis_refs)
# for n in range(0,pxp.dim):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
    # pxp.keys[pxp.basis_refs[n]] = n
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
# pxp_syms = model_sym_data(pxp,[translational_general(pxp,2),PT(pxp)])

# H = Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = np.array([[0,1],[1,0]])
# H.model = np.array([[0,1,1,1,0]])
# H.model_coef = np.array([1])
# H.gen()
# H.sector.find_eig()

# ent = entropy(pxp)
# ent_vals = np.zeros(pxp.dim)
# for n in range(0,np.size(ent_vals,axis=0)):
    # ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
# plt.scatter(H.sector.eigvalues(),ent_vals)
# plt.show()

# k=pxp_syms.find_k_ref(z.ref)
# eigvectors = np.zeros(pxp.dim)
# U = dict()
# eigvalues = []
# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])
    # U[n] = pxp_syms.basis_transformation(k[n])
    # temp = np.dot(U[n],H.sector.eigvectors(k[n]))
    # eigvectors = np.vstack((eigvectors,np.transpose(temp)))
    # eigvalues = np.append(eigvalues,H.sector.eigvalues(k[n]))
# eigvectors = np.transpose(np.delete(eigvectors,0,axis=0))
# fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
# plt.show()

# print("Dim = "+str(np.size(H.sector.matrix(k[0]),axis=0)))
# ls = level_stats(H.sector.eigvalues(k[0]))
# rstat = ls.mean()
# print(ls.mean())

# np.save("pxxxp,ls,"+str(pxp.N),rstat)

# ent = entropy(pxp)
# ent_vals = np.zeros(np.size(eigvectors,axis=1))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(ent_vals,axis=0))):
    # ent_vals[n] = ent.eval(eigvectors[:,n])

# Hp = Hamiltonian(pxp,pxp_syms)
# Hp.site_ops[1] = np.array([[0,0],[1,0]])
# Hp.site_ops[2] = np.array([[0,1],[0,0]])
# Hp.model = np.array([[0,1,2,1,0],[0,2,1,2,0],[0,2,1,2,0],[0,1,2,1,0]])
# Hp.model_coef = np.array([1,1,1,1])
# Hp.uc_size = np.array([4,4,4,4])
# Hp.uc_pos = np.array([2,3,0,1])
# Hp.gen()
# H.gen()
# from Calculations import gen_fsa_basis
# fsa_basis = gen_fsa_basis(Hp.sector.matrix(),z.prod_basis(),pxp.N)
# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# efsa,ufsa = np.linalg.eigh(H_fsa)
# u_comp = np.dot(fsa_basis,ufsa)
# ent_fsa = np.zeros(np.size(u_comp,axis=1))
# for n in range(0,np.size(ent_fsa,axis=0)):
    # ent_fsa[n] = ent.eval(u_comp[:,n])

# plt.scatter(eigvalues,ent_vals)
# plt.scatter(efsa,ent_fsa,marker="x",color="red",s=100)
# plt.show()
# np.save("pxxxp,e,"+str(pxp.N),eigvalues)
# np.save("pxxxp,ent,"+str(pxp.N),ent_vals)
# np.save("pxxxp,fsa,e,"+str(pxp.N),efsa)
# np.save("pxxxp,fsa,ent,"+str(pxp.N),ent_fsa)
