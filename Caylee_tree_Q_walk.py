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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

L=50
k=2
H=np.power(k,0.5)*(np.diag(np.ones(L-1),1)+np.diag(np.ones(L-1),-1))
e,u = np.linalg.eigh(H)

z=np.zeros(np.size(H,axis=0))
z[0] = 1

z_energy = u[0,:]
t=np.arange(0,100,0.01)
f=np.zeros(np.size(t))
from Calculations import time_evolve_state
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.title("Caylee Tree, L="+str(L)+", branching ratio="+str(k))
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle 0(t) \vert 0(0) \rangle \vert^2$")
plt.show()
    
