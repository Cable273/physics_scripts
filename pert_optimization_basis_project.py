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

N_vals = np.arange(10,18,2)
exact_coef = np.zeros(np.size(N_vals))
fsa_coef = np.zeros(np.size(N_vals))
perm_coef = np.zeros(np.size(N_vals))
cube_coef = np.zeros(np.size(N_vals))
pbar=ProgressBar()
for count in pbar(range(0,np.size(N_vals,axis=0))):
    N = N_vals[count]
    pxp = unlocking_System([0],"periodic",2,N)
    pxp.gen_basis()
    pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

    H0=spin_Hamiltonian(pxp,"x",pxp_syms)

    V1=Hamiltonian(pxp,pxp_syms)
    V1.site_ops[1] = np.array([[0,1],[1,0]])
    # V1.model = np.array([[0,1,1,1,0]])
    V1.model = np.array([[0,0,1,0],[0,1,0,0]])
    V1.model_coef = np.array([1,1])

    H0.gen()
    V1.gen()

    def fidelity_eval(psi_energy,e,t):
        evolved_state = time_evolve_state(psi_energy,e,t)
        f = np.abs(np.vdot(evolved_state,psi_energy))**2
        return -f

    def pert_opt_fidelity(coef,basis=None,plot=False):
        H = H_operations.add(H0,V1,np.array([1,coef]))
        psi=zm_state(2,1,pxp).prod_basis()
        if basis is None:
            H = H.sector.matrix()
        else:
            H = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
            psi = np.dot(np.conj(np.transpose(basis)),psi)
            psi = psi / np.power(np.vdot(psi,psi),0.5)
        e,u = np.linalg.eigh(H)
        psi_energy = np.dot(np.conj(np.transpose(u)),psi)
        if plot is True:
            t=np.arange(0,20,0.01)
            f=np.zeros(np.size(t))
            for n in range(0,np.size(f,axis=0)):
                f[n] = -fidelity_eval(psi_energy,e,t[n])
            plt.plot(t,f)
            plt.show()

        res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(4.6,5.5))
        f_max = -fidelity_eval(psi_energy,e,res.x)
        print(f_max,coef,res.x)
        return -f_max

    fsa_basis = np.load("./basis_data/fsa_basis,"+str(pxp.N)+".npy")
    perm_basis = np.load("./basis_data/perm_basis,"+str(pxp.N)+".npy")
    cube_basis = np.load("./basis_data/Subcube_basis,"+str(pxp.N)+".npy")

    from scipy.optimize import minimize,minimize_scalar
    res_exact = minimize_scalar(lambda coef: pert_opt_fidelity(coef),method="golden",bracket=(0.1,0.13))
    res_fsa = minimize_scalar(lambda coef: pert_opt_fidelity(coef,basis=fsa_basis),method="golden",bracket=(0.1,0.13))
    res_perm = minimize_scalar(lambda coef: pert_opt_fidelity(coef,basis=perm_basis),method="golden",bracket=(0.1,0.13))
    res_cube = minimize_scalar(lambda coef: pert_opt_fidelity(coef,basis=cube_basis),method="golden",bracket=(0.1,0.13))

    exact_coef[count] = res_exact.x
    fsa_coef[count] = res_fsa.x
    perm_coef[count] = res_perm.x
    cube_coef[count] = res_cube.x
# np.save("./optimal_pert_coef/pxxxp,exact_coef",exact_coef)
# np.save("./optimal_pert_coef/pxxxp,fsa_coef",fsa_coef)
# np.save("./optimal_pert_coef/pxxxp,perm_coef",perm_coef)
# np.save("./optimal_pert_coef/pxxxp,cube_coef",cube_coef)

np.save("./optimal_pert_coef/ppxp,exact_coef",exact_coef)
np.save("./optimal_pert_coef/ppxp,fsa_coef",fsa_coef)
np.save("./optimal_pert_coef/ppxp,perm_coef",perm_coef)
np.save("./optimal_pert_coef/ppxp,cube_coef",cube_coef)
