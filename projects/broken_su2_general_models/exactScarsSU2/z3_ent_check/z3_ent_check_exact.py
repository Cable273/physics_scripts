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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,gen_fsa_basis

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

#init system
N=18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

Hp = Hamiltonian(pxp,pxp_syms)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,2,0],[0,1,0],[0,1,0]])
Hp.model_coef = np.array([1,1,1])
Hp.uc_size = np.array([3,3,3])
Hp.uc_pos = np.array([2,0,1])

Hp.gen()
Hm = Hp.herm_conj()
H = H_operations.add(Hp,Hm,np.array([1,1]))

z=zm_state(3,1,pxp)
H.sector.find_eig()
overlap = eig_overlap(z,H).eval()
from Calculations import get_top_band_indices
scar_indices = get_top_band_indices(H.sector.eigvalues(),overlap,int(2*pxp.N/3),100,200,e_diff=0.5)

plt.scatter(H.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

ent_vals = np.zeros(pxp.dim)
ent = entropy(pxp)
pbar=ProgressBar()
for n in pbar(range(0,np.size(ent_vals,axis=0))):
    ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
plt.scatter(H.sector.eigvalues(),ent_vals)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],ent_vals[scar_indices[n]],marker="x",color="red",s=200)
plt.show()
