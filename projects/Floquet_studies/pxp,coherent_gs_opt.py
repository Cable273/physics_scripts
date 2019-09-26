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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

#choose
import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#create coherent states to init trajectories from
def coherent_state(alpha,su2_states,J):
    wf_su2 = np.zeros(np.size(su2_states,axis=1),dtype=complex)

    eta = alpha*np.tan(np.abs(alpha))/np.abs(alpha)
    for m in range(0,np.size(wf_su2,axis=0)):
        wf_su2[m] = np.power(ncr(int(2*J),m),0.5)*np.power(eta,m)

    wf = wf_su2[0] * su2_states[:,0]
    for m in range(1,np.size(wf_su2,axis=0)):
        wf = wf + wf_su2[m] * su2_states[:,m]

    wf = wf / np.power(1+np.abs(eta)**2,J)
    return wf

from Calculations import time_evolve_state

def cell_no(state_energy,eigs,T,delta_t):
    t_range = np.arange(0,T,delta_t)
    lin = 1-t_range/T

    fid = np.zeros(np.size(t_range))
    for n in range(0,np.size(t_range,axis=0)):
        evolved_state = time_evolve_state(state_energy,eigs,t_range[n])
        fid[n] = np.abs(np.vdot(evolved_state,state_energy))**2
    integrand = np.multiply(lin,fid)
    integral = np.trapz(integrand,t_range)
    phase_space_cells = T/(2*integral)
    return phase_space_cells

from Calculations import time_evolve_state
def trajectory(init_state,t_range,H,system):
    state_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),init_state)
    x0 = np.vdot(init_state,np.dot(X.sector.matrix(k),init_state))
    y0 = np.vdot(init_state,np.dot(Y.sector.matrix(k),init_state))
    z0 = np.vdot(init_state,np.dot(Z.sector.matrix(k),init_state))

    x = np.zeros(np.size(t_range))
    y = np.zeros(np.size(t_range))
    z = np.zeros(np.size(t_range))
    f = np.zeros(np.size(t_range))
    for n in range(0,np.size(t_range,axis=0)):
        evolved_state = time_evolve_state(state_energy,H.sector.eigvalues(k),t_range[n])
        x[n] = np.real(np.vdot(evolved_state,np.dot(X_energy,evolved_state)))/J
        y[n] = np.real(np.vdot(evolved_state,np.dot(Y_energy,evolved_state)))/J
        z[n] = np.real(np.vdot(evolved_state,np.dot(Z_energy,evolved_state)))/J
        f[n] = np.abs(np.vdot(evolved_state,state_energy))**2
    return x,y,z,f

def project_to_subspace(state,system1,system2):
    # projected_state = np.zeros(np.size(system2.basis_refs),dtype=complex)
    projected_state = np.copy(state)
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>1e-5:
            # if system1.basis_refs[n] in system2.basis_refs:
                # projected_state[system2.keys[system1.basis_refs[n]]] = state[n]
            if system1.basis_refs[n] not in system2.basis_refs:
                projected_state[n] = 0
    projected_state = projected_state / np.power(np.vdot(projected_state,projected_state),0.5)
    return projected_state
        

pxp = unlocking_System("pxp",[0],"periodic",2,24)
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
#looking at coherent states, lin combinations of |J,m>, |J,m> = (sum_n s_n^+)^m |000....>
#ie |J,m> lin combs of zero momenta states, only need to consider zero momenta
k=[0]
H.gen(k)
H.sector.find_eig(k)
mom_refs = pxp_syms.find_block_refs(k)
mom_keys = dict()
for n in range(0,np.size(mom_refs,axis=0)):
    mom_keys[mom_refs[n]] = n

# operators for coherent states
X = spin_Hamiltonian(pxp,"x",pxp_syms)
X.gen(k)
Y = spin_Hamiltonian(pxp,"y",pxp_syms)
Y.gen(k)
Z = spin_Hamiltonian(pxp,"z",pxp_syms)
Z.gen(k)
sp = X.sector.matrix(k)+1j*Y.sector.matrix(k)
sm = X.sector.matrix(k)-1j*Y.sector.matrix(k)
spe,spu = np.linalg.eig(sp)
sme,smu = np.linalg.eig(sm)

X_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(X.sector.matrix(k),H.sector.eigvectors(k)))
Y_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(Y.sector.matrix(k),H.sector.eigvectors(k)))
Z_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(Z.sector.matrix(k),H.sector.eigvectors(k)))

#create basis of total SU(2) states with S_max. Create by acting on fully polarized with sm_full
state = np.zeros(np.size(mom_refs))
state[0] = 1
su2_states = state

s=1/2*(pxp.base-1)
J = int(pxp.N*s)
for n in range(0,J):
    state = np.dot(sp,state)
    state = state / np.power(np.vdot(state,state),0.5)
    su2_states = np.vstack((su2_states,state))
su2_states = np.transpose(su2_states)

# print(su2 states)
# for n in range(0,np.size(su2_states,axis=1)):
    # print("State "+str(n))
    # for m in range(0,np.size(su2_states[:,n],axis=0)):
        # if np.abs(su2_states[m,n])>1e-5:
            # # print(su2_states[m,n],pxp.basis[m])
            # print(su2_states[m,n],int_to_bin_base_m(mom_refs[m],pxp.base,pxp.N))
        
gs = H.sector.eigvectors(k)[:,0]

# def cost(params):
    # alpha = params[0] + 1j * params[1]
    # state = coherent_state(alpha,su2_states,J)
    # return -np.abs(np.vdot(gs,state))**2

# from scipy.optimize import minimize
# res = minimize(lambda params: cost(params),method="Powell",x0=[1,1])
# print(res)

def cost(alpha):
    state = coherent_state(alpha,su2_states,J)
    return -np.abs(np.vdot(gs,state))**2

from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda alpha: cost(alpha),method="Golden")
print(res)
