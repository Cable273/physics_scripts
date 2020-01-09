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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,gen_fsa_basis

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0 = Hamiltonian(pxp)
H0.site_ops[1] = np.array([[0,1],[1,0]])
H0.model = np.array([[0,1,1,1,0]])
H0.model_coef = np.array([1])

V = Hamiltonian(pxp)
V.site_ops[1] = np.array([[0,1],[1,0]])
V.site_ops[4] = np.array([[0,0],[0,1]])
V.model = np.array([[0,1,1,1,0,4,0],[0,4,0,1,1,1,0],[0,1,1,1,0,4,0],[0,4,0,1,1,1,0]])
V.model_coef = np.array([1,1,1,1])
V.uc_size = np.array([4,4,4,4])
V.uc_pos = np.array([2,0,0,2])

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

from copy import deepcopy
from Hamiltonian_Classes import H_operations
from scipy.optimize import minimize,minimize_scalar
def fidelity_error(coef,plot=False):
    coef = coef[0]
    H = H_operations.add(H0,V,np.array([1,coef]))
    H = H.sector.matrix()
    e,u = np.linalg.eigh(H)
    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())


    if plot is True:
        t=np.arange(0,20,0.01)
        f=np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            f[n] = -fidelity_eval(psi_energy,e,t[n])
        plt.plot(t,f)
        plt.show()
        
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(4.5,5.5))
    f = -fidelity_eval(psi_energy,e,res.x)
    print(coef,f)
    if res.x <1e-5:
        return 1000
    else:
        return -f

H0.gen()
V.gen()

from scipy.optimize import minimize
res = minimize(lambda coef: fidelity_error(coef),method="Nelder-Mead",x0=[0])
fidelity_error([res.x],plot=True)
