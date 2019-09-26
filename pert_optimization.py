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

N = 10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

# H0=clock_Hamiltonian(pxp,pxp_syms)
H0=spin_Hamiltonian(pxp,"x",pxp_syms)

V1=Hamiltonian(pxp,pxp_syms)
V1.site_ops[1] = np.array([[0,1],[0,0]])
V1.site_ops[2] = np.array([[0,0],[1,0]])
V1.model = np.array([[0,0,1,2,0],[0,0,2,1,0],[0,2,1,0,0],[0,1,2,0,0]])
V1.model_coef = np.array([1,-1,1,-1])

H0.gen()
V1.gen()
plt.matshow(np.abs(V1.sector.matrix()))
plt.show()

z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
# H0.gen(k[0])
# V1.gen(k[0])
# V2.gen(k[0])
# H0.gen(k[1])
# V1.gen(k[1])
# V2.gen(k[1])

def fidelity_eval(z,H,t):
    f=fidelity(z,H).eval([t],z)
    # f=fidelity(z,H,"use sym").eval([t],z)
    # evolved_state = time_evolve_state(z_energy,e,t)
    # f= np.abs(np.vdot(z_energy,evolved_state))**2
    return -f

from scipy.optimize import minimize,minimize_scalar
z=zm_state(2,1,pxp)
def pert_opt_fidelity(coef,plot=False):
    H = H_operations.add(H0,V1,np.array([1,coef]))
    # H = H_operations.add(H,V2,np.array([1,coef[1]]))
    z=zm_state(2,1,pxp)
    # for n in range(0,np.size(k,axis=0)):
        # print(np.shape(H.sector.matrix(k[n])))
        # H.sector.find_eig(k[n])
    H.sector.find_eig()
    if plot is True:
        fidelity(z,H).plot(np.arange(0,20,0.01),z)
        plt.show()
    # z_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
    res = minimize_scalar(lambda t: fidelity_eval(z,H,t),method="golden",bracket=(4.6,5.5))
    # f_max = -fidelity_eval(z_energy,H.sector.eigvalues(),res.x)
    f_max = -fidelity_eval(z,H,res.x)
    print(f_max,coef,res.x)
    return -f_max

def pert_opt_equidistant_scars(coef,print_e_diff=False,plot=False):
    from Calculations import get_top_band_indices
    H = H_operations.add(H0,V,np.array([1,coef]))
    H.sector.find_eig(k[0])
    H.sector.find_eig(k[1])
    e = np.append(H.sector.eigvalues(k[0]),H.sector.eigvalues(k[1]))
    o1 = eig_overlap(z,H,k[0]).eval()
    o2 = eig_overlap(z,H,k[1]).eval()
    overlap = np.append(o1,o2)
    e,overlap = (list(t) for t in zip(*sorted(zip(e,overlap))))
    scar_indices = np.unique(np.sort(get_top_band_indices(e,overlap,N)))

    if plot==True:
        plt.scatter(e,overlap)
        for n in range(0,np.size(scar_indices,axis=0)):
            plt.scatter(e[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
        plt.show()

    e_scars = np.zeros(np.size(scar_indices))
    for n in range(0,np.size(e_scars,axis=0)):
        # e_scars[n] = H.sector.eigvalues()[scar_indices[n]]
        e_scars[n] = e[scar_indices[n]]
    e_diff = np.zeros(np.size(e_scars)-1)
    for n in range(0,np.size(e_diff,axis=0)):
        e_diff[n] = e_scars[n+1]-e_scars[n]

    #rescale so first diff is 1
    e_diff = 1/e_diff[0]*e_diff
    #cost function: sum of mean pairwise differences
    cost = 0
    for n in range(0,np.size(e_diff,axis=0)):
        temp = []
        for m in range(0,np.size(e_diff,axis=0)):
            if m != n:
                temp = np.append(temp,np.abs(e_diff[n]-e_diff[m]))
        cost = cost + np.mean(temp)
    # for n in range(0,np.size(e_diff,axis=0)):
        # cost = cost + np.abs(1-e_diff[n])
    print(coef,cost)
    if print_e_diff == True:
        print(e_diff)
    return cost
        
def pert_opt_level_stats(coef):
    H = H_operations.add(H0,V,np.array([1,coef]))
    H.sector.find_eig([0,0])
    ls = level_stats(H.sector.eigvalues([0,0]))
    r = ls.mean()
    print(coef,r)
    return r

# coef = np.arange(-0.2,0.2,0.001)
# level_vals = np.zeros(np.size(coef))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(coef,axis=0))):
    # level_vals[n] = pert_opt_level_stats(coef[n])
# plt.axvline(x=0.108256)
# plt.plot(coef,level_vals)
# plt.xlabel(r"$\lambda$")
# plt.ylabel(r"$\langle r \rangle$")
# plt.title(r"$PXP+\lambda(PXPP+PPXP) \, \textrm{Level Stats}, N=20$")
# plt.show()
    
# res = minimize_scalar(lambda coef: pert_opt_fidelity(coef),method="golden",bracket=(-0.02,-0.01))
from scipy.optimize import minimize
res = minimize_scalar(lambda coef: pert_opt_fidelity(coef),method="golden",bracket=(-0.01,-0.1))
# res = minimize(lambda coef: pert_opt_fidelity(coef),method="powell",x0=[0.1,0.1])
print(res.x)
# pert_opt_fidelity(res.x,plot=True)
# # res = minimize_scalar(lambda coef: pert_opt_fidelity(coef),method="golden",bracket=(0.09,0.1))
# # res = minimize_scalar(lambda coef: pert_opt_equidistant_scars(coef),method="golden",bracket=(0.01,0.1))
# # res = minimize(lambda coef: pert_opt_level_stats(coef),method="powell",x0=[0.02])

# # H=H_operations.add(H0,V,np.array([1,res.x]))
# # H=H_operations.add(H0,V,np.array([1,0.108256]))
# # H=H_operations.add(H0,V,np.array([1,0.051]))
# # H=H_operations.add(H0,V,np.array([1,0.024]))
# # H=H_operations.add(H0,V,np.array([1,res.x]))

# print("\n")
# coef_f = 0.12284863
# # coef_e_diff = 0.1789254
# coef_e_diff = res.x
# print("Optimized for fidelity")
# pert_opt_equidistant_scars(coef_f,print_e_diff=True,plot=True)
# print("\n")
# print("Optimized for delta E")
# pert_opt_equidistant_scars(coef_e_diff,print_e_diff=True,plot=True)
# # H_e_diff=H_operations.add(H0,V,np.array([1,0.1805648742]))
# # H_f=H_operations.add(H0,V,np.array([1,0.12284863]))
# # H_e_diff.sector.find_eig()
# # H_f.sector.find_eig()
# # z=zm_state(2,1,pxp)

# # overlap_f = eig_overlap(z,H_f).eval()
# # overlap_e_diff = eig_overlap(z,H_e_diff).eval()

# # scar_indices_f = np.unique(np.sort(get_top_band_indices(H_f.sector.eigvalues(),overlap_f,N)))
# # scar_indices_e_diff = np.unique(np.sort(get_top_band_indices(H_e_diff.sector.eigvalues(),overlap_e_diff,N)))

# # plt.scatter(H_f.sector.eigvalues(),overlap_f)
# # for n in range(0,np.size(scar_indices_f,axis=0)):
    # # plt.scatter(H_f.sector.eigvalues()[scar_indices_f[n]],overlap_f[scar_indices_f[n]],marker="x",color="red",s=100)
# # plt.show()

# # plt.scatter(H_e_diff.sector.eigvalues(),overlap_e_diff)
# # for n in range(0,np.size(scar_indices_f,axis=0)):
    # # plt.scatter(H_e_diff.sector.eigvalues()[scar_indices_e_diff[n]],overlap_e_diff[scar_indices_e_diff[n]],marker="x",color="red",s=100)
# # plt.show()

# # e_scars_f = np.zeros(np.size(scar_indices_f))
# # e_scars_e_diff = np.zeros(np.size(scar_indices_e_diff))
# # for n in range(0,np.size(e_scars_f,axis=0)):
    # # e_scars_f[n] = H_f.sector.eigvalues()[scar_indices_f[n]]

# # for n in range(0,np.size(e_scars_e_diff,axis=0)):
    # # e_scars_e_diff[n] = H_e_diff.sector.eigvalues()[scar_indices_e_diff[n]]

# # e_scars_f_diff = np.zeros(np.size(e_scars_f)-1)
# # e_scars_e_diff_diff = np.zeros(np.size(e_scars_e_diff)-1)

# # for n in range(0,np.size(e_scars_f_diff,axis=0)):
    # # e_scars_f_diff[n] = e_scars_f[n+1]-e_scars_f[n]

# # for n in range(0,np.size(e_scars_e_diff_diff,axis=0)):
    # # e_scars_e_diff_diff[n] = e_scars_e_diff[n+1]-e_scars_e_diff[n]

# # e_scars_f_diff = 1/e_scars_f_diff[0]*e_scars_f_diff
# # e_scars_e_diff_diff = 1/e_scars_e_diff_diff[0]*e_scars_e_diff_diff
# # c1 = 0
# # for n in range(0,np.size(e_scars_f_diff,axis=0)):
    # # c1 = c1 + np.abs(1-e_scars_f_diff[n])

# # c2 = 0
# # for n in range(0,np.size(e_scars_f_diff,axis=0)):
    # # c2 = c2 + np.abs(1-e_scars_e_diff_diff[n])
# # print("\n")
# # print("Optimized for fidelity")
# # print(c1)
# # print(e_scars_f_diff)
# # print("\n")

# # print("Optimized for energy diff")
# # print(c2)
# # print(e_scars_e_diff_diff)

# ## dynamics
# # z=zm_state(2,1,pxp)
# # k=pxp_syms.find_k_ref(z.ref)
# # for n in range(0,np.size(k,axis=0)):
    # # H.sector.find_eig(k[n])
    # # eig_overlap(z,H,k[n]).plot()
# # plt.show()
# # fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
# # plt.show()

# # ls = level_stats(H.sector.eigvalues(k[0]))
# # print(ls.mean())
