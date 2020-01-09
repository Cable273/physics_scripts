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

N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.model = np.array([[0,1,1,1,0]])
H.model_coef = np.array([1])

Hp = Hamiltonian(pxp)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,2,1,2,0],[0,1,2,1,0]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([4,4])
Hp.uc_pos = np.array([0,2])

H.gen()
Hp.gen()

z=zm_state(4,1,pxp)
from Calculations import gen_fsa_basis
from Diagnostics import print_wf
fsa_basis = gen_fsa_basis(Hp.sector.matrix(),z.prod_basis(),pxp.N)
for n in range(0,np.size(fsa_basis,axis=1)):
    print("\n")
    print_wf(fsa_basis[:,n],pxp,1e-2)

H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e_fsa, u_fsa = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)

H.sector.find_eig()
eig_overlap(z,H).plot()
plt.scatter(e_fsa,overlap_fsa,marker="x",s=100,color="red")
plt.show()

exact_overlap = np.zeros(np.size(fsa_basis,axis=1))
u_fsa_comp = np.dot(fsa_basis,u_fsa)
for n in range(0,np.size(u_fsa_comp,axis=1)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.plot(e_fsa,exact_overlap,marker="s")
plt.show()

        
    
    

