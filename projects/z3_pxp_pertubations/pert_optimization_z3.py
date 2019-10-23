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
pxp_syms=model_sym_data(pxp,[translational_general(pxp,order=3)])

z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

V1 = Hamiltonian(pxp,pxp_syms)
V1.site_ops[1] = np.array([[0,1],[1,0]])
V1.model = np.array([[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0]])
V1.model_coef = np.array([1,1,1,1])
V1.uc_size = np.array([3,3,3,3])
V1.uc_pos = np.array([1,2,2,1])

V2 = Hamiltonian(pxp,pxp_syms)
V2.site_ops[1] = np.array([[0,1],[1,0]])
V2.model = np.array([[0,0,1,0],[0,1,0,0]])
V2.model_coef = np.array([1,1])
V2.uc_size = np.array([3,3])
V2.uc_pos = np.array([0,0])

V3 = Hamiltonian(pxp,pxp_syms)
V3.site_ops[1] = np.array([[0,1],[1,0]])
V3.model = np.array([[0,1,1,1,0],[0,1,1,1,0]])
V3.model_coef = np.array([1,1])
V3.uc_size = np.array([3,3])
V3.uc_pos = np.array([0,2])

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()
V1.gen()
V2.gen()
V3.gen()

def fidelity_eval(psi_energy,e,t):
    # f=fidelity(z,H).eval([t],z)
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(psi_energy,evolved_state))**2
    # f=fidelity(z,H,"use sym").eval([t],z)
    # evolved_state = time_evolve_state(z_energy,e,t)
    # f= np.abs(np.vdot(z_energy,evolved_state))**2
    return -f

from scipy.optimize import minimize,minimize_scalar
def pert_opt_fidelity(coef,plot=False):
    # H = H_operations.add(H0,V1,np.array([1,coef[0]]))
    H = H_operations.add(H0,V1,np.array([1,coef]))
    # H = H_operations.add(H,V2,np.array([1,coef[1]]))
    # H = H_operations.add(H,V3,np.array([1,coef[2]]))
    z=zm_state(3,1,pxp)
    psi = z.prod_basis()
    # psi = np.load("./z3,entangled_MPS_coef,15.npy")
    H.sector.find_eig()
    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),psi)
    if plot is True:
        eig_overlap(z,H).plot()
        plt.show()
        t=np.arange(0,20,0.01)
        f=np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            f[n] = -fidelity_eval(psi_energy,H.sector.eigvalues(),t[n])
        plt.plot(t,f)
        plt.show()
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(2.5,5))
    # f_max = -fidelity_eval(z_energy,H.sector.eigvalues(),res.x)
    f_max = -fidelity_eval(psi_energy,H.sector.eigvalues(),res.x)
    print(coef,f_max,res.x)
    if np.abs(res.x)<1e-5:
        return 1000
    else:
        return -f_max

from scipy.optimize import minimize_scalar
# res = minimize(lambda coef: pert_opt_fidelity(coef,plot=True),method="powell",x0=[0.18243653,-0.10390499,0.054452])
# res = minimize(lambda coef: pert_opt_fidelity(coef),method="powell",x0=[0.18243653,-0.10390499,0.054452])
coef_vals = np.arange(0,0.5,0.01)
pbar=ProgressBar()
f_max = np.zeros(np.size(coef_vals))
for n in pbar(range(0,np.size(f_max,axis=0))):
    f_max[n] = pert_opt_fidelity(coef_vals[n],plot=True)
plt.plot(coef_vals,f_max)
plt.show()
    
# for count in range(0,np.size(array,axis=0)):
    
# res = minimize_scalar(lambda coef: pert_opt_fidelity(coef),method="golden",bracket=(0.1,0.3))
# print(res.x)
# # pert_opt_fidelity([res.x],plot=True)
# pert_opt_fidelity(res.x,plot=True)

