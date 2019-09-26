#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sparse_linalg
import sys
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,get_top_band_indices

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

pert_coef = 0.051
H0=spin_Hamiltonian(pxp,"x",pxp_syms)
V = Hamiltonian(pxp,pxp_syms)
V.site_ops[1] = np.array([[0,1],[1,0]])
V.site_ops[2] = np.array([[-1,0],[0,1]])
V.model = np.array([[2,0,1,0],[0,1,0,2]])
V.model_coef = np.array([1,1])
V.gen()
H0.gen()
H=H_operations.add(H0,V,np.array([1,-pert_coef]))

#find rescaling such that E1-E0 = 1
def gap(H,c):
    e,u = np.linalg.eigh(c*H)
    gap = e[1]-e[0]
    # print(gap,c)
    return gap

from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda c: np.abs(gap(H.sector.matrix(),c)-1),method="golden", bracket=(0.5,1.5))
c=res.x
H=H_operations.add(H0,V,np.array([c,-c*pert_coef]))
H.sector.find_eig()
print(H.sector.eigvalues()[1]-H.sector.eigvalues()[0])

#P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.model = np.array([[0,1,0]])
pe.model_coef = np.array([1])
pe.gen(parity=0)

#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=1)

#ZP+P, + on even sites
zPpPe = Hamiltonian(pxp,pxp_syms)
zPpPe.site_ops[1] = np.array([[0,1],[0,0]])
zPpPe.site_ops[2] = np.array([[-1,0],[0,1]])
zPpPe.model = np.array([[2,0,1,0]])
zPpPe.model_coef = np.array([1])
zPpPe.gen(parity=1)

#P+PZ, + on even sites
PpPze = Hamiltonian(pxp,pxp_syms)
PpPze.site_ops[1] = np.array([[0,1],[0,0]])
PpPze.site_ops[2] = np.array([[-1,0],[0,1]])
PpPze.model = np.array([[0,1,0,2]])
PpPze.model_coef = np.array([1])
PpPze.gen(parity=0)

#ZP-P, - on odd sites
zPmPo = Hamiltonian(pxp,pxp_syms)
zPmPo.site_ops[1] = np.array([[0,0],[1,0]])
zPmPo.site_ops[2] = np.array([[-1,0],[0,1]])
zPmPo.model = np.array([[2,0,1,0]])
zPmPo.model_coef = np.array([1])
zPmPo.gen(parity=0)

#P-PZ, - on odd sites
PmPzo = Hamiltonian(pxp,pxp_syms)
PmPzo.site_ops[1] = np.array([[0,0],[1,0]])
PmPzo.site_ops[2] = np.array([[-1,0],[0,1]])
PmPzo.model = np.array([[0,1,0,2]])
PmPzo.model_coef = np.array([1])
PmPzo.gen(parity=1)

#check hams working right
# temp0 = ref_state(1,pxp)
# temp1 = zm_state(2,1,pxp)
# test0 = np.dot(pe.sector.matrix(),temp0.prod_basis())
# test1 = np.dot(PpPze.sector.matrix(),temp0.prod_basis())
# test2= np.dot(zPpPe.sector.matrix(),temp0.prod_basis())
# test3 = np.dot(mo.sector.matrix(),temp1.prod_basis())
# test4 = np.dot(zPmPo.sector.matrix(),temp1.prod_basis())
# test5= np.dot(PmPzo.sector.matrix(),temp1.prod_basis())
# from Diagnostics import print_wf
# print(temp0.bits)
# print("Pe")
# print_wf(test0,pxp,1e-1)
# print("\n")
# print("PpPze")
# print_wf(test1,pxp,1e-1)
# print("\n")
# print("zPpPe")
# print_wf(test2,pxp,1e-1)
# print("\n")
# print(temp1.bits)
# print("mo")
# print_wf(test3,pxp,1e-1)
# print("\n")
# print("zPmPo")
# print_wf(test4,pxp,1e-1)
# print("\n")
# print("PmPzo")
# print_wf(test5,pxp,1e-1)

Hp = H_operations.add(pe,mo,np.array([c,c]))
Hp = H_operations.add(Hp,zPpPe,np.array([1,-c*pert_coef]))
Hp = H_operations.add(Hp,PpPze,np.array([1,-c*pert_coef]))
Hp = H_operations.add(Hp,zPmPo,np.array([1,-c*pert_coef]))
Hp = H_operations.add(Hp,PmPzo,np.array([1,-c*pert_coef]))

Hp = Hp.sector.matrix()
Hm = np.conj(np.transpose(Hp))

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = com(Hp,Hm)

z=zm_state(2,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,pxp.N):
    new_state = np.dot(Hm,current_state)
    new_state = new_state / np.power(np.vdot(new_state,new_state),0.5)
    fsa_basis = np.vstack((fsa_basis,new_state))
    current_state = new_state

fsa_basis = np.transpose(fsa_basis)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
Hz_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hz,fsa_basis))

print(np.diag(Hz_fsa))
print(Hz_fsa[1,1]-Hz_fsa[0,0])

plt.matshow(np.abs(H_fsa))
plt.show()

plt.matshow(np.abs(Hz_fsa))
plt.show()

overlap = eig_overlap(z,H).eval()
scar_indices = np.unique(np.sort(get_top_band_indices(H.sector.eigvalues(),overlap,pxp.N)))

plt.scatter(H.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

Z_eff = np.zeros((np.size(scar_indices),np.size(scar_indices)))
for n in range(0,np.size(scar_indices,axis=0)):
    Z_eff[n,n] = H.sector.eigvalues()[scar_indices[n]]
# print(np.sort(np.diag(Z_eff)))
# print(Z_eff[1,1]-Z_eff[0,0])

fsa_z_e_diffs = np.zeros(np.size(Hz_fsa,axis=0)-1)
Z_eff_e_diffs = np.zeros(np.size(Hz_fsa,axis=0)-1)
for n in range(0,np.size(fsa_z_e_diffs,axis=0)):
    fsa_z_e_diffs[n] = Hz_fsa[n+1,n+1]-Hz_fsa[n,n]
    Z_eff_e_diffs[n] = Z_eff[n+1,n+1]-Z_eff[n,n]
#rescale fsa_z_e_diffs so delta E_0 = 1
fsa_z_e_diffs = 1/fsa_z_e_diffs[0]*fsa_z_e_diffs

print(np.diag(Z_eff))
print(np.diag(Hz_fsa))
print(np.diag(Z_eff)[0]/np.diag(Hz_fsa)[0]*np.diag(Hz_fsa))
# print(Z_eff[0]/Hz_fsa[0]*Hz_fsa)

# plt.plot(fsa_z_e_diffs,label="FSA HZ (with pert)")
# plt.plot(Z_eff_e_diffs,label="Z_eff (with pert)")
# plt.xlabel(r"$n$")
# plt.ylabel(r"$E_{n+1}-E_n$")
# plt.title(r"Eigenvalue differences, alternative SU(2) Z Definitions, $PXP+\lambda(ZPXP+PXPZ)$")
# plt.show()
    

# S = 1/2*(np.size(Z_eff,axis=0)-1)

# def com(a,b):
    # return np.dot(a,b)-np.dot(b,a)
# #pauli ops of su(2) subspace
# m = np.arange(-S,S)
# couplings = np.power(S*(S+1)-m*(m+1),0.5)
# sP = np.diag(couplings,1)
# sM = np.diag(couplings,-1)
# Y = (sP - sM)/(2j)
# X = (sP + sM)/(2)
# Z = com(X,Y)

# e,u = np.linalg.eigh(Y)
# d=0.01
# theta_vals = np.arange(0,2*math.pi+d,d)
# F_vals = np.zeros(np.size(theta_vals))
# # for n in range(0,np.size(theta_vals,axis=0)):
# theta = math.pi/2
# # theta = theta_vals[n]
# # D = np.diag(np.exp(1j * theta * e))
# # R = np.dot(np.conj(u),np.dot(D,u))
# R = sp.linalg.expm(-1j*theta*Y)

# X_eff = np.dot(np.transpose(np.conj(R)),np.dot(Z_eff,R))
# # for n in range(0,np.size(X_eff,axis=0)):
    # # for m in range(0,np.size(X_eff,axis=0)):
        # # if np.abs(X_eff[n,m])>1e-5:
            # # print(X_eff[n,m],n,m)
    
# # plt.matshow(np.abs(X_eff))
# # plt.show()
# # diff = X-X_eff
# # F_norm = np.power(np.trace(np.dot(diff,np.conj(np.transpose(diff)))),0.5)
# # F_vals[n] = F_norm
# # print(np.min(F_vals))
# # plt.plot(theta_vals,F_vals)
# # plt.axvline(x=math.pi/2)
# # plt.xlabel(r"$\theta$")
# # plt.ylabel(r"$\vert X-X_{eff} \vert_F$")
# # plt.title(r"Comparing exact pauli $X (2S+1 = 2N+1)$ rep, to approximate $X_{eff}$")
# # plt.show()

# # theta =math.pi/2
# # R = sp.linalg.expm(-1j*theta*Y)
# # X_eff = np.dot(np.transpose(np.conj(R)),np.dot(Z_eff,R))
# # plt.matshow(np.abs(X_eff))
# # plt.show()

# # print("X eff")
# # for n in range(0,np.size(X_eff,axis=0)):
# # for m in range(0,np.size(X_eff,axis=0)):
    # # temp = np.abs(X_eff[n,m])
    # # if temp > 1e-5:
        # # print(temp,n,m)

