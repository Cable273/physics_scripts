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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spinx_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection

k=0.1
I=0.1

trajectories = dict()
no_trajectories = 300
pbar=ProgressBar()
for traject_index in pbar(range(0,no_trajectories)):
    t=np.arange(0,20,0.01)
    coords = np.zeros((np.size(t),2))
    coords[0,0] = np.random.uniform(0,2*math.pi)
    coords[0,1] = np.random.uniform(0,1)
    for n in range(1,np.size(coords,axis=0)):
        coords[n,1] = coords[n-1,1] + k * np.sin(coords[n-1,0])
        coords[n,0] = (coords[n-1,0] + coords[n,1]/I) % (2*math.pi)

    trajectories[traject_index] = coords
    plt.scatter(coords[:,0],coords[:,1],s=0.05)
plt.xlabel(r"$\phi$")
plt.ylabel(r"$P$")
plt.title(r"Kicked Rotor, I="+str(I)+", k="+str(k))
plt.show()
