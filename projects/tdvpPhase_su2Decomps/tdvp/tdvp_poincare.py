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
def M1(x):
    return -4 * (3+np.cos(2*x[1]) - 2 * np.cos(2*x[2])*np.sin(x[2])**2)
def M2(x):
    return -64 * (np.cos(x[2])**2 + np.sin(x[0])**2 * np.sin(x[2])**2)
def M3(x):
    return 16*(1-np.cos(x[1])**2*np.sin(x[0])**2)

def x1_dot(x):
    temp = np.sin(x[1])*(4*np.sin(x[1])*np.cos(3*x[2])) +2*np.sin(x[0])*np.sin(2*x[1])*(-2*np.cos(x[0])**2 * np.cos(2*x[2])+np.cos(2*x[0])-3) -2 * (3*np.cos(2*x[1])+5)*np.cos(x[2])
    return temp/M1(x)

def x2_dot(x):
    temp = -4*np.sin(2*x[2])*((np.cos(2*x[0])+7)*np.sin(x[1]) - 2 * np.sin(x[0])**2*np.sin(3*x[1])) - 8 * np.cos(x[0]) * (4*np.cos(x[0])**2 * np.cos(2*x[2]) + 5) + 8 * np.cos(3*x[0])
    return temp/M2(x)

def x3_dot(x):
    temp = -2*np.sin(x[1])**2*(np.sin(2*x[0])*np.sin(3*x[2])) + np.sin(2*x[0])*(np.cos(2*x[1])+7)*np.sin(x[2])+8*np.cos(2*x[0])*np.cos(x[1])**3 + 10 * np.cos(x[1]) - 2* np.cos(3*x[1])
    return temp/M3(x)

def f(x):
    fx = x1_dot(x)
    fy = x2_dot(x)
    fz = x3_dot(x)
    return np.array([fx,fy,fz])

from RK4 import RK4
rk = RK4(f,3)

def find_crossings(psi_init,surfaceNormal_index,delta_t_init):
    newDelta_t = delta_t_init / 10
    rk.evolve(psi_init,0,delta_t_init+newDelta_t,newDelta_t)
    newTrajectory = rk.y_out % (2*math.pi)
    for n in range(0,np.size(newTrajectory,axis=0)-1):
        if newTrajectory[n][surfaceNormal_index] > math.pi and newTrajectory[n+1][surfaceNormal_index] < math.pi:
            psiOut = newTrajectory[n]
            break
    return psiOut

#specifically for 3 param phase space of 3 site unit cell PXP TDVP
def poincare(initState,surfaceNormal_index,t_max,delta_t,tol):
    #project initState onto poincare surface (plane)
    initState[surfaceNormal_index] = 0

    #rough RK4 to obtain time series
    rk.evolve(initState,0,t_max,delta_t)
    trajectories = rk.y_out % (2*math.pi)

    #identify sign changes of surface normal coordinates
    poincare_points = np.zeros(3)
    for n in range(0,np.size(trajectories,axis=0)-1):
        if trajectories[n][surfaceNormal_index] > math.pi and trajectories[n+1][surfaceNormal_index] < math.pi:
            #hone in to exact crossing
            temp_timeStep = delta_t
            psi_out = trajectories[n]
            for m in range(0,10):
                psi_out = find_crossings(psi_out,surfaceNormal_index,temp_timeStep)
                temp_timeStep = temp_timeStep / 10
            poincare_points = np.vstack((poincare_points,psi_out))
    poincare_points = np.delete(poincare_points,0,axis=0)
    return poincare_points

    
temp = poincare(np.random.uniform(0,2*math.pi,3),1,1000,0.01,tol=1e-5)
plt.scatter(temp[:,0],temp[:,2])
plt.show()
# gridSize = 0.5
# pbar=ProgressBar()
# for n in pbar(np.arange(0,math.pi+gridSize,gridSize)):
    # for m in np.arange(0,math.pi+gridSize,gridSize):
        # temp = poincare(np.array([n,0,m]),1,100,0.1,tol=1e-5)
        # plt.scatter(temp[:,2],temp[:,0])
# plt.xlabel(r"$\theta_3$")
# plt.ylabel(r"$\theta_1$")
# plt.show()
