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

y0 = np.array([0,math.pi/2,0])
from RK4 import RK4
rk = RK4(f,3)
rk.evolve(y0,0,20,0.01)
print(rk.y_out)
rk.plot()
np.save("z3,tdvp,angles",rk.y_out)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x_i(t)$")
# plt.title(r"TDVP Angle Evolution")
# plt.show()
# np.savetxt("tdvp,z3,angle_evolution",rk.y_out)

# angles = rk.y_out
# N=12
# #convert MPS -> wf array
# system = unlocking_System([0],"periodic",2,N)
# system.gen_basis()

# from Calculations import mps_uc3_angles2wf
# z=zm_state(3,1,system)
# f=np.zeros(np.size(angles,axis=0))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(angles,axis=0))):
    # wf = mps_uc3_angles2wf(angles[n],system)
    # f[n] = np.abs(np.vdot(wf,z.prod_basis()))**2
# plt.plot(rk.t,f)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
# plt.title(r"$PXP, \vert Z_3 \rangle$ TDVP Evolution, N="+str(N))
# plt.show()
