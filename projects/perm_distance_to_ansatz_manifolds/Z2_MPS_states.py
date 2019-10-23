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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
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

N = 10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

#generate perm basis
def perm_keys(n,m,N):
    temp = np.array([n,m])
    key = bin_to_int_base_m(temp,int(N/2)+1)
    return key
def perm_inv_key(ref):
    bits = int_to_bin_base_m(ref,int(N/2)+1,2)
    return bits

perm_basis_states = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    A_occ = 0
    B_occ = 0
    for m in range(0,N):
        if m % 2 == 0:
            A_occ += pxp.basis[n][m]
        else:
            B_occ += pxp.basis[n][m]

    key = perm_keys(A_occ,B_occ,pxp.N)
    if key in list(perm_basis_states.keys()):
        perm_basis_states[key][n] += 1
    else:
        perm_basis_states[key] = np.zeros(pxp.dim)
        perm_basis_states[key][n] += 1

basis_refs = list(perm_basis_states.keys())
#normalize basis states
for n in range(0,np.size(basis_refs,axis=0)):
    perm_basis_states[basis_refs[n]] = perm_basis_states[basis_refs[n]] / np.power(np.vdot(perm_basis_states[basis_refs[n]],perm_basis_states[basis_refs[n]]),0.5)

perm_basis = np.zeros(pxp.dim)
for n in range(0,np.size(basis_refs,axis=0)):
    perm_basis = np.vstack((perm_basis,perm_basis_states[basis_refs[n]]))
perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))
perm_dim = np.size(perm_basis,axis=1)


H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

H_perm = np.dot(np.conj(np.transpose(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
e,u = np.linalg.eigh(H_perm)

#time evolve state in perm basis,
z_ref = perm_keys(int(N/2),0,N)
z_index = find_index_bisection(z_ref,basis_refs)
psi_energy = np.conj(u[z_index,:])

t=np.arange(0,20,0.1)
evolved_states = np.zeros((pxp.dim,np.size(t)),dtype=complex)
for n in range(0,np.size(t,axis=0)):
    evolved_state_perm_energy = time_evolve_state(psi_energy,e,t[n])
    evolved_state_perm = np.dot(u,evolved_state_perm_energy)
    evolved_states_product = np.dot(perm_basis,evolved_state_perm)
    evolved_states[:,n] = evolved_states_product
    print(np.vdot(evolved_states_product,evolved_states_product))

system = unlocking_System([0],"periodic",2,N)
system.gen_basis()
#minimize distance of evolved states with tensor tree ansatz
def TT_wf(theta1,phi1,theta2,phi2):
    def A_up(theta,phi):
        return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])

    def A_down(theta,phi):
        return np.array([[np.cos(theta),0],[np.sin(theta),0]])

    A_ups = dict()
    A_downs = dict()
    A_ups[0] = A_up(theta1,phi1)
    A_ups[1] = A_up(theta2,phi2)

    A_downs[0] = A_down(theta1,phi1)
    A_downs[1] = A_down(theta2,phi2)

    tensors = dict()
    K = 2
    for n in range(0,K):
        tensors[n] = np.zeros((2,np.size(A_ups[0],axis=0),np.size(A_ups[0],axis=1)),dtype=complex)
    tensors[0][0] = A_downs[0]
    tensors[0][1] = A_ups[0]

    tensors[1][0] = A_downs[1]
    tensors[1][1] = A_ups[1]

    # tensors[2][0] = A_downs[2]
    # tensors[2][1] = A_ups[2]

    from MPS import periodic_MPS
    psi = periodic_MPS(N)
    for n in range(0,N,1):
        psi.set_entry(n,tensors[int(n%2)],"both")

    #convert MPS -> wf array
    wf = np.zeros(system.dim,dtype=complex)
    for n in range(0,np.size(system.basis_refs,axis=0)):
        bits = system.basis[n]
        coef = psi.node[0].tensor[bits[0]]
        for m in range(1,np.size(bits,axis=0)):
            coef = np.dot(coef,psi.node[m].tensor[bits[m]])
        coef = np.trace(coef)
        wf[n] = coef
    return wf

def TT_wf_distance(psi,tt_params):
    psi_tt = TT_wf(tt_params[0],tt_params[1],tt_params[2],tt_params[3])
    diff = psi - psi_tt
    return np.real(np.vdot(diff,diff))

wf = TT_wf(0,0,math.pi/2,0)
from Diagnostics import print_wf
print_wf(wf,pxp,1e-2)
from scipy.optimize import minimize
TT_distance = np.zeros(np.size(t))
pbar=ProgressBar()
for n in pbar(range(0,np.size(evolved_states,axis=1))):
    if n == 0:
        res = minimize(lambda tt_params: TT_wf_distance(evolved_states[:,n],tt_params),method="nelder-mead",x0=[0,0,math.pi/2,0])
        last_coef = res.x
    else:
        res = minimize(lambda tt_params: TT_wf_distance(evolved_states[:,n],tt_params),method="nelder-mead",x0=last_coef)
        last_coef = res.x
    # print(res.x,TT_wf_distance(evolved_states[:,n],res.x))
    TT_distance[n] = TT_wf_distance(evolved_states[:,n],res.x)
np.save("mps_distance,t,10",t)
np.save("mps_distance,distance,10",TT_distance)
print(TT_distance[0])
plt.plot(t,TT_distance)
plt.show()
