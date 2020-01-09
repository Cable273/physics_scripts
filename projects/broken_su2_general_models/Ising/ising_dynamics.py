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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
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

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp = pxp.U1_sector(5)
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

J_orth = 1
Jz = 1

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,0],[1,0]])
H.site_ops[2] = np.array([[0,1],[0,0]])
H.site_ops[3] = np.array([[-1/2,0],[0,1/2]])
H.model = np.array([[0,1,2,0],[0,2,1,0],[0,3,3,0]])
H.model_coef = np.array([J_orth/2,J_orth/2,Jz])

H.gen()
H.sector.find_eig()
pbar=ProgressBar()
for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    z= ref_state(pxp.basis_refs[n],pxp)
    fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.title(r"$H = P( J(XX+YY)+J_z ZZ)P$"+"\n"+r"Largest sector computational basis quenches, $N=$"+str(pxp.N))
plt.show()
# z= ref_state(1,pxp)
# fidelity(z,H).plot(np.arange(0,20,0.01),z)
# plt.show()

# ent = entropy(pxp)
# ent_vals = np.zeros(pxp.dim)
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(ent_vals,axis=0))):
    # ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
# plt.scatter(H.sector.eigvalues(),ent_vals)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$S$")
# plt.title(r"$H = P(XX+YY)P$"+"\n"+r"Largest sector Entropy, $N=$"+str(pxp.N))
# plt.show()
