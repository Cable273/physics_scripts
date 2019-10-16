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

N = 15
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,3),PT(pxp)])

#generate perm basis
def perm_keys(n,m,u,N):
    temp = np.array([n,m,u])
    key = bin_to_int_base_m(temp,int(N/3)+1)
    return key
def perm_inv_key(ref):
    bits = int_to_bin_base_m(ref,int(N/3)+1,3)
    return bits

perm_basis_states = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    A_occ = 0
    B_occ = 0
    C_occ = 0
    for m in range(0,N):
        if m % 3 == 0:
            A_occ += pxp.basis[n][m]
        elif m % 3 == 1:
            B_occ += pxp.basis[n][m]
        else:
            C_occ += pxp.basis[n][m]

    key = perm_keys(A_occ,B_occ,C_occ,pxp.N)
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
z=zm_state(3,1,pxp)
# psi_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
psi = np.dot(np.conj(np.transpose(perm_basis)),z.prod_basis())
psi = psi/np.power(np.vdot(psi,psi),0.5)
psi_energy = np.dot(np.conj(np.transpose(u)),psi)

t=np.arange(0,10,0.1)
evolved_states = np.zeros((pxp.dim,np.size(t)),dtype=complex)
for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(),t[n])
    # evolved_prod_basis = np.dot(H.sector.eigvectors(),evolved_state)
    evolved_perm_energy_basis = time_evolve_state(psi_energy,e,t[n])
    evolved_perm_basis= np.dot(u,evolved_perm_energy_basis)
    evolved_prod_basis = np.dot(perm_basis,evolved_perm_basis)
    evolved_states[:,n] = evolved_prod_basis

system = unlocking_System([0],"periodic",2,N)
system.gen_basis()
#minimize distance of evolved states with tensor tree ansatz
def mps_wf(theta1,phi1,theta2,phi2,theta3,phi3):
    def A_up(theta,phi):
        return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])

    def A_down(theta,phi):
        return np.array([[np.cos(theta),0],[np.sin(theta),0]])

    #create MPs

    A_ups = dict()
    A_downs = dict()
    A_ups[0] = A_up(theta1,phi1)
    A_ups[1] = A_up(theta2,phi2)
    A_ups[2] = A_up(theta3,phi3)

    A_downs[0] = A_down(theta1,phi1)
    A_downs[1] = A_down(theta2,phi2)
    A_downs[2] = A_down(theta3,phi3)

    tensors = dict()
    K = 3
    for n in range(0,K):
        tensors[n] = np.zeros((2,np.size(A_ups[0],axis=0),np.size(A_ups[0],axis=1)),dtype=complex)
    tensors[0][0] = A_downs[0]
    tensors[0][1] = A_ups[0]

    tensors[1][0] = A_downs[1]
    tensors[1][1] = A_ups[1]

    tensors[2][0] = A_downs[2]
    tensors[2][1] = A_ups[2]

    from MPS import periodic_MPS
    psi = periodic_MPS(N)
    for n in range(0,N,1):
        psi.set_entry(n,tensors[int(n%3)],"both")

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

def mps_wf_distance(psi,mps_params,print=False):
    psi_tt = mps_wf(mps_params[0],mps_params[1],mps_params[2],mps_params[3],mps_params[4],mps_params[5])
    diff = psi - psi_tt
    if print is True:
        print_wf(psi_tt,pxp,1e-2)
    return np.real(np.vdot(diff,diff))
wf = mps_wf(0,0,math.pi/2,0,0,0)
from Diagnostics import print_wf
print_wf(wf,pxp,1e-2)

from scipy.optimize import minimize
mps_distance = np.zeros(np.size(t))
pbar=ProgressBar()
for n in pbar(range(0,np.size(evolved_states,axis=1))):
    if n == 0:
        res = minimize(lambda mps_params: mps_wf_distance(evolved_states[:,n],mps_params),method="powell",x0=[0,0,math.pi/2,0,0,0])
        last_coef = res.x
        # print_wf(evolved_states[:,n],pxp,1e-2)
        # print("\n")
        # mps_wf_distance(evolved_states[:,n],res.x,print=True)
    else:
        res = minimize(lambda mps_params: mps_wf_distance(evolved_states[:,n],mps_params),method="powell",x0=last_coef)
        last_coef = res.x
    mps_distance[n] = mps_wf_distance(evolved_states[:,n],res.x)
# print(mps_distance[0])
plt.plot(t,mps_distance)
plt.show()


    




