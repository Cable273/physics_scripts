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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state
from copy import deepcopy

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

N=14
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp = pxp.U1_sector([int(N/2)])
print(pxp.dim)
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[1,2],[2,1]])
Hp[0].model_coef = np.array([1,1])
Hp[0].uc_size = np.array([2,2])
Hp[0].uc_pos = np.array([0,1])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].site_ops[3] = np.array([[-1,0],[0,1]])
Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
Hp[1].model = np.array([[0,1,2],[4,1,2,],[1,2,0],[1,2,4],[0,2,1],[4,2,1],[2,1,0],[2,1,4]])
Hp[1].model_coef = np.array([1,1,1,1,1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2,2,2,2,2])
Hp[1].uc_pos = np.array([1,1,0,0,0,0,1,1])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1,0],[0,1]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[1,3,3,2],[2,3,3,1]])
Hp[2].model_coef = np.array([1,1])
Hp[2].uc_size = np.array([2,2])
Hp[2].uc_pos = np.array([0,1])

z=zm_state(2,1,pxp)
z2 = zm_state(2,1,pxp,1)

k=[0,0]
block_refs = pxp_syms.find_block_refs(k)
if z.ref in block_refs:
    neel_loc = find_index_bisection(z.ref,block_refs)
else:
    neel_loc = find_index_bisection(z2.ref,block_refs)

for n in range(0,len(Hp)):
    Hp[n].gen(k)

psi_mom = np.zeros(np.size(block_refs))
psi_mom[neel_loc] = 1

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return f

from scipy.optimize import minimize_scalar,minimize
def fidelity_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
        
    Hm_total = Hp_total.herm_conj()

    H = H_operations.add(Hp_total,Hm_total,np.array([1,1]))
    H.sector.find_eig(k)

    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),psi_mom)

    t=np.arange(0,20,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(k),t[n])
        f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.5:
            cut = n
            break
    f_max = np.max(f[cut:])
    error = 1-f_max
        

    # res = minimize_scalar(lambda t: -fidelity_eval(psi_energy,H.sector.eigvalues(k),t),method="golden")
    # f0 = fidelity_eval(psi_energy,H.sector.eigvalues(k),res.x)
    # error = 1-f0
    print(coef,error)
    # if (np.abs(coef)>1).any():
        # return 1000
    # else:
    return error

coef = np.load("./xy,pert_coef,10.npy")
# coef = np.zeros(2)
print(coef)
res = minimize(lambda coef: fidelity_error(coef),method="powell",x0=coef)
print(res.x)
coef = res.x
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    
Hp = Hp_total
Hm = Hp.herm_conj()

H = H_operations.add(Hp,Hm,np.array([1,1]))
H.sector.find_eig(k)

psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),psi_mom)
overlap = np.abs(psi_energy)**2
plt.scatter(H.sector.eigvalues(k),overlap)
plt.show()

t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(k),t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.show()


