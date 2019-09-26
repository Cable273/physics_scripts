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

N = 15
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational_general(pxp,order=3)])

z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,0,0]])
V1_ops[0].model_coef = np.array([1])
V1_ops[0].gen(uc_size=3,uc_pos=1)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[1].model = np.array([[0,0,1,0]])
V1_ops[1].model_coef = np.array([1])
V1_ops[1].gen(uc_size=3,uc_pos=2)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
V1_ops[2].gen(uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
V1_ops[3].gen(uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,0,1,0]])
V2_ops[0].model_coef = np.array([1])
V2_ops[0].gen(uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,0,0]])
V2_ops[1].model_coef = np.array([1])
V2_ops[1].gen(uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,1,1,1,0]])
V3_ops[0].model_coef = np.array([1])
V3_ops[0].gen(uc_size=3,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,1,1,0]])
V3_ops[1].model_coef = np.array([1])
V3_ops[1].gen(uc_size=3,uc_pos=2)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()


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
    H = H_operations.add(H0,V1,np.array([1,coef[0]]))
    H = H_operations.add(H,V2,np.array([1,coef[1]]))
    H = H_operations.add(H,V3,np.array([1,coef[2]]))
    # z=zm_state(3,1,pxp)
    psi = np.load("./z3,entangled_MPS_coef,15.npy")
    H.sector.find_eig()
    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),psi)
    if plot is True:
        t=np.arange(0,20,0.01)
        f=np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            f[n] = -fidelity_eval(psi_energy,H.sector.eigvalues(),t[n])
        plt.plot(t,f)
        plt.show()
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(4.6,5.5))
    # f_max = -fidelity_eval(z_energy,H.sector.eigvalues(),res.x)
    f_max = -fidelity_eval(psi_energy,H.sector.eigvalues(),res.x)
    print(f_max,coef,res.x)
    if np.abs(res.x)<1e-5:
        return 1000
    else:
        return -f_max

from scipy.optimize import minimize
res = minimize(lambda coef: pert_opt_fidelity(coef),method="powell",x0=[0.18243653,-0.10390499,0.054452])
print(res.x)
pert_opt_fidelity(res.x,plot=True)

