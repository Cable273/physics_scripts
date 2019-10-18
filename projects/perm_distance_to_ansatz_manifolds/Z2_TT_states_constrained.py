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

N = 18
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

t=np.arange(0,10,0.1)
evolved_states = np.zeros((pxp.dim,np.size(t)),dtype=complex)
for n in range(0,np.size(t,axis=0)):
    evolved_state_perm_energy = time_evolve_state(psi_energy,e,t[n])
    evolved_state_perm = np.dot(u,evolved_state_perm_energy)
    evolved_state_prod = np.dot(perm_basis,evolved_state_perm)
    evolved_states[:,n] = evolved_state_prod

#minimize distance of evolved states with tensor tree ansatz
def TT_wf(theta1,phi1,theta2,phi2):
    c1_up = 1j * np.exp(-1j * phi1)*np.tan(theta2)
    c1_down = np.cos(theta1)

    c2_up = 1j * np.exp(-1j*phi2)*np.tan(theta1)
    c2_down = np.cos(theta2)
    wf = np.zeros(pxp.dim,dtype=complex)

    for n in range(0,np.size(pxp.basis,axis=0)):
        A_occ = 0
        B_occ = 0
        for m in range(0,pxp.N):
            if m % 2 == 0:
                A_occ += pxp.basis[n][m]
            else:
                B_occ += pxp.basis[n][m]

        coef = np.power(c1_down,N/2-A_occ)*np.power(c1_up,A_occ)*np.power(c2_down,N/2-B_occ)*np.power(c2_up,B_occ)
        wf[n] = coef
    wf = wf / np.power(np.vdot(wf,wf),0.5)
    return wf

def TT_wf_distance(psi,tt_params):
    psi_tt = TT_wf(tt_params[0],tt_params[1],tt_params[2],tt_params[3])
    diff = psi - psi_tt
    return np.real(np.vdot(diff,diff))

wf = TT_wf(math.pi/4,0,math.pi/4,0)
from Diagnostics import print_wf
print_wf(wf,pxp,1e-2)
from scipy.optimize import minimize
TT_distance = np.zeros(np.size(t))
pbar=ProgressBar()
for n in pbar(range(0,np.size(evolved_states,axis=1))):
    if n == 0:
        res = minimize(lambda tt_params: TT_wf_distance(evolved_states[:,n],tt_params),method="powell",x0=[math.pi/4,0,math.pi/4,0])
        print(res.x)
        last_coef = res.x
    else:
        res = minimize(lambda tt_params: TT_wf_distance(evolved_states[:,n],tt_params),method="powell",x0=last_coef)
        last_coef = res.x
    TT_distance[n] = TT_wf_distance(evolved_states[:,n],res.x)
np.save("tt_full,t",t)
np.save("tt_full,distance",TT_distance)
print(TT_distance[0])
plt.plot(t,TT_distance)
plt.show()


    




