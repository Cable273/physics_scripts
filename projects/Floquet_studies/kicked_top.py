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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

tau=2.1
alpha=2

fig = plt.figure()
ax = Axes3D(fig)
trajectories = dict()
no_trajectories = 10
pbar=ProgressBar()
t=np.arange(0,10,0.01)
for traject_index in pbar(range(0,no_trajectories)):
    coords = np.zeros((np.size(t),3))
    coords_polar = np.zeros((np.size(t),2))
    coords[0,0] = np.random.uniform(-1,1)
    coords[0,1] = np.random.uniform(-1,1)
    coords[0,2] = np.power(1-coords[0,0]**2-coords[0,1]**2,0.5)
    x=np.random.uniform(0,1)
    if x>0.5:
        coords[0,2] = - coords[0,2]

    for n in range(1,np.size(coords,axis=0)):
        tempx = coords[n-1,0]
        tempy = coords[n-1,1]*np.cos(alpha)-coords[n-1,2]*np.sin(alpha)
        tempz = coords[n-1,1]*np.sin(alpha)+coords[n-1,2]*np.cos(alpha)

        coords[n,0] = tempx * np.cos(tau*tempz) - tempy * np.sin(tau*tempz)
        coords[n,1] = tempx * np.sin(tau*tempz) + tempy * np.cos(tau*tempz)
        coords[n,2] = tempz 

    coords_polar[:,0] = np.arccos(coords[:,2]) 
    coords_polar[:,1] = np.arctan(coords[:,1]/coords[:,0])

    trajectories[traject_index] = coords_polar
    # plt.scatter(coords_polar[:,0],coords_polar[:,1],s=0.05)
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],s=1)
# print(coords[:,0])
# print(coords[:,1])
# print(coords[:,2])
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\phi$")
plt.title(r"Kicked Top, $\tau=$"+str(tau)+r", $\alpha=$"+str(alpha))
plt.show()
