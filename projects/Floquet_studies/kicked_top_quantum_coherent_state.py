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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection

import operator as op
from functools import reduce

def is_zero(a,tol):
    return (np.abs(a)<tol).all()
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)


#choose
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

pxp = unlocking_System("pxp",[0,1],"periodic",150 ,1)
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])

#drive variables
# phi = -0.5*(1-np.power(5,0.5))
# tau = 0.5*phi
tau = 0.1
alpha = 0.89
j = 1/2*(pxp.base-1)
print(j)

Jx=spin_Hamiltonian(pxp,"x").site_ops[1]
Jy=spin_Hamiltonian(pxp,"y").site_ops[1]
Jz=spin_Hamiltonian(pxp,"z").site_ops[1]

exp_Jz2 = np.diag(np.exp(-1j*alpha/(2*j)*np.diag(np.dot(Jz,Jz))))
ex,ux = np.linalg.eigh(Jx)
exp_Jx = np.dot(ux,np.dot(np.diag(np.exp(-1j*tau*ex)),np.conj(np.transpose(ux))))

F = np.dot(exp_Jz2,exp_Jx)
print("Finding Floquet Spectra")
e,u = np.linalg.eig(F)
print("Done")

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def trajectory(z_init,F):
    zn1 = z_init
    zn = z_init

    f = np.zeros(np.size(t))
    x_exp = np.zeros(np.size(t))
    y_exp = np.zeros(np.size(t))
    z_exp = np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        f[n] = np.abs(np.vdot(zn1,zn))**2
        x_exp[n] = np.real(np.vdot(zn1,np.dot(Jx,zn1)))
        y_exp[n] = np.real(np.vdot(zn1,np.dot(Jy,zn1)))
        z_exp[n] = np.real(np.vdot(zn1,np.dot(Jz,zn1)))
        # print(x_exp[0],y_exp[0],z_exp[0])

        zn1 = np.dot(F,zn1)
        zn1 = zn1/ np.power(np.vdot(zn1,zn1),0.5)
    return x_exp,y_exp,z_exp,f

sp = Jx+1j*Jy
sm = Jx-1j*Jy
print("Finding ladder eigenvectors")
spe,spu = np.linalg.eig(sp)
sme,smu = np.linalg.eig(sm)

#create coherent states to init trajectories from
def coherent_state(alpha,su2_states):
    N = np.size(su2_states,axis=1)
    J = 1/2*(N-1)
    wf_su2 = np.zeros(np.size(su2_states,axis=1),dtype=complex)

    eta = alpha*np.tan(np.abs(alpha))/np.abs(alpha)
    for m in range(0,np.size(wf_su2,axis=0)):
        wf_su2[m] = np.power(ncr(int(2*J),m),0.5)*np.power(eta,m)

    wf = wf_su2[0] * su2_states[:,0]
    for m in range(1,np.size(wf_su2,axis=0)):
        wf = wf + wf_su2[m] * su2_states[:,m]

    wf = wf / np.power(1+np.abs(eta)**2,J)
    return wf

#create basis of total SU(2) states with S_max. Create by acting on fully polarized with sm_full
su2_states=np.zeros(np.size(pxp.basis_refs))
su2_states[0] = 1
temp = su2_states
for n in range(0,pxp.base-1):
    temp = np.dot(sm,temp)
    temp = temp / np.power(np.vdot(temp,temp),0.5)
    su2_states = np.vstack((su2_states,temp))
su2_states = np.transpose(su2_states)

N= np.size(su2_states,axis=1)
J = 1/2*(N-1)
        
fig = plt.figure()
ax = Axes3D(fig)
no_traj = 100
states = dict()
print("Plotting trajectories of coherent states")
t= np.arange(0,300)
pbar=ProgressBar()
# vals = np.arange(-3,3,0.01)
for n in pbar(range(0,no_traj)):
# for n in pbar(range(0,np.size(vals,axis=0))):
    a = np.random.uniform(0,2*math.pi)
    b = np.random.uniform(0,2*math.pi)
    states[n] = coherent_state(a+1j*b,su2_states)
    # states[n] = coherent_state(a,b,su2_states)

    x,y,z,f = trajectory(states[n],F)
    x = x / J
    y = y / J
    z = z / J
    theta = np.arccos(z) 
    phi = np.arctan(y/x)
    # plt.scatter(theta,phi,s=1)
    ax.scatter(x,y,z,s=1)
plt.show()

cState_truncated = []
vals2 = []
pbar=ProgressBar()
for n in pbar(range(0,len(states))):
    norm = np.vdot(states[n],states[n])
    if is_zero(norm-1,1e-8):
        vals2 = np.append(vals2,vals[n])
        x = np.real(np.vdot(states[n],np.dot(Jx,states[n])))/J
        y = np.real(np.vdot(states[n],np.dot(Jy,states[n])))/J
        z = np.real(np.vdot(states[n],np.dot(Jz,states[n])))/J
        theta = np.arccos(z) 
        phi = np.arctan(y/x)

        #calculate percent of states to throw away in floquet eigenbasis, locate regular regions
        cState_floquet_eigbasis = np.dot(np.conj(u),states[n])
        wf_probs = np.sort(np.abs(cState_floquet_eigbasis)**2)
        #finding no. of states to truncate to exhaust norm within 1%
        S0 = np.sum(wf_probs)
        for n in range(1,np.size(wf_probs,axis=0)):
            S=np.sum(wf_probs[n:])
            diff = S0-S
            percent = diff/S0*100
            if percent >=1:
                truncated_no = n
                break
        cState_truncated = np.append(cState_truncated,truncated_no)
plt.plot(vals2,np.log(cState_truncated)/np.log(j))
plt.show()
