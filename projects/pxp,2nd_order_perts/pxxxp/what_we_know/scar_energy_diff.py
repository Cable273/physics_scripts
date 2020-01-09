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

N=22
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp)])

#create Hamiltonian
# a=0.122959959
# a=0.108
# # a=0
# H = Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = np.array([[0,1],[1,0]])
# H.model = np.array([[1],[0,0,1,0],[0,1,0,0]])
# # H.model = np.array([[1],[0,1,1,1,0]])
# H.model_coef = np.array([1,a,a])

# z=zm_state(2,1,pxp)
# k=pxp_syms.find_k_ref(z.ref)
# overlap = dict()
# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])
    # overlap[n] = eig_overlap(z,H,k[n]).eval()

# e = H.sector.eigvalues(k[0])
# e = np.append(e,H.sector.eigvalues(k[1]))

# overlap_total =overlap[0]
# overlap_total = np.append(overlap_total,overlap[1])

# to_del=[]
# for n in range(0,np.size(overlap_total,axis=0)):
    # if overlap_total[n] <-10:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap_total=np.delete(overlap_total,to_del[n])
    # e=np.delete(e,to_del[n])
    
# np.save("e_temp",e)
# np.save("overlap_temp",overlap_total)
e = np.load("./e_temp.npy")
overlap_total = np.load("./overlap_temp.npy")
e,overlap_total = (list(t) for t in zip(*sorted(zip(e,overlap_total))))
plt.scatter(e,overlap_total)

from Calculations import get_top_band_indices
scar_indices = np.sort(get_top_band_indices(e,overlap_total,pxp.N+2,350,400,e_diff=1))
scar_indices = np.append(scar_indices,1)

scar_indices = np.append(scar_indices,np.size(e)-2)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(e[scar_indices[n]],overlap_total[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

scar_e = np.zeros(np.size(scar_indices))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_e[n] = e[scar_indices[n]]
scar_e = np.sort(scar_e)

scar_e_diff = np.zeros(np.size(scar_e)-1)
for n in range(0,np.size(scar_e_diff,axis=0)):
    scar_e_diff[n] = scar_e[n+1] - scar_e[n]
np.save("pxp,pxxxp,scar_e_diff,ppxp,"+str(pxp.N),scar_e_diff)
# np.save("pxp,pxxxp,scar_e_diff,no_pert,"+str(pxp.N),scar_e_diff)
plt.plot(scar_e_diff)
plt.show()
