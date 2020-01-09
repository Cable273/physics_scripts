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
    return exp(Q2,psi) - exp(Q,psi)**2

N=10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0 = spin_Hamiltonian(pxp,"x")
V = Hamiltonian(pxp)
V.site_ops[1] = np.array([[0,1],[1,0]])
V.model = np.array([[0,1,1,1,0]])
V.model_coef = np.array([1])
# V.model = np.array([[0,1,0,0],[0,0,1,0]])
# V.model_coef = np.array([1,1])
H0.gen()

V.gen()

from Calculations import gen_krylov_basis
def krylov_su2_coupling_error(coef):
    coef = coef[0]
    H = H_operations.add(H0,V,np.array([1,coef]))

    z=zm_state(2,1,pxp)
    krylov_basis = gen_krylov_basis(H.sector.matrix(),pxp.N,z.prod_basis(),pxp,orth="qr")
    H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
    couplings = np.diag(H_krylov,1)

    s = pxp.N/4
    m = np.arange(-s,s)
    su2_couplings_half = np.power(s*(s+1)-m*(m+1),0.5)

    s = pxp.N/2
    m = np.arange(-s,s)
    su2_couplings_full = np.power(s*(s+1)-m*(m+1),0.5)

    coupling_diff_full = couplings - su2_couplings_full
    error = np.power(np.abs(np.vdot(coupling_diff_full,coupling_diff_full)),0.5)
    # error = np.abs(couplings[3] - su2_couplings_half[3])
    print(coef,error)
    return error

from scipy.optimize import minimize
res = minimize(lambda coef: krylov_su2_coupling_error(coef),method="Nelder-Mead",x0=[0.122])
print(res.x)
