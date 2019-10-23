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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
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

N_vals = np.load("./N.npy")
f_vals = np.zeros(np.size(N_vals))
coef_vals = np.load("./ppxp,optimal_coef.npy")
def f_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(psi_energy,evolved_state))**2
    return -f

for count in range(0,np.size(N_vals,axis=0)):
    N = N_vals[count]
    pxp = unlocking_System([0],"periodic",2,N)
    pxp.gen_basis()
    pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

    H=Hamiltonian(pxp,pxp_syms)
    H.site_ops[1] = np.array([[0,1],[1,0]])
    H.model = np.array([[0,1,0],[0,0,1,0],[0,1,0,0]])
    a=coef_vals[count]
    H.model_coef = np.array([1,a,a])
    H.gen()
    H.sector.find_eig()

    z=zm_state(2,1,pxp)
    psi_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t: f_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(4.5,5.5))
    f_max = -f_eval(psi_energy,H.sector.eigvalues(),res.x)
    f_vals[count] = f_max
    print(f_max)
np.save("ppxp,f_max",f_vals)

