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
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
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

N = 12
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1/2],[1/2,0]])
H.site_ops[2] = np.array([[1,0],[0,0]])
H.site_ops[3] = np.array([[0,0],[0,1]])
H.model = np.array([[2,1,2],[2,1,3],[3,1,2]])
V=30e-1
H.model_coef = np.array([V,1,1])
# H.model = np.array([[0,1]])
# H.model_coef = np.array([1])

k=[0]
H.gen(k)
H.sector.find_eig(k)

block_refs = pxp_syms.find_block_refs(k)
block_keys = dict()
for n in range(0,np.size(block_refs,axis=0)):
    block_keys[block_refs[n]] = n

neel=zm_state(2,1,pxp,1)
pol = ref_state(0,pxp)
all_ones = bin_state(np.append([0],np.ones(pxp.N-1)),pxp)

neel_trans = np.zeros(np.size(block_refs))
pol_trans = np.zeros(np.size(block_refs))
all_ones_trans = np.zeros(np.size(block_refs))

neel_trans[block_keys[neel.ref]] = 1
pol_trans[block_keys[pol.ref]] = 1
all_ones_trans[block_keys[all_ones.ref]] = 1

neel_trans_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),neel_trans)
pol_trans_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),pol_trans)
all_ones_trans_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),all_ones_trans)

t=np.arange(0,80,0.01)
f_neel = np.zeros(np.size(t))
f_pol = np.zeros(np.size(t))
f_all_ones = np.zeros(np.size(t))

for n in range(0,np.size(t,axis=0)):
    evolved_neel = time_evolve_state(neel_trans_energy,H.sector.eigvalues(k),t[n])
    # evolved_pol = time_evolve_state(pol_trans_energy,H.sector.eigvalues(k),t[n])
    # evolved_all_ones = time_evolve_state(all_ones_trans_energy,H.sector.eigvalues(k),t[n])

    f_neel[n] = np.abs(np.vdot(evolved_neel,neel_trans_energy))**2
    f_pol[n] = np.abs(np.vdot(evolved_neel,pol_trans_energy))**2
    f_all_ones[n] = np.abs(np.vdot(evolved_neel,all_ones_trans_energy))**2

plt.plot(t,f_neel,label=r"$\vert \phi \rangle = \vert Z_2,k=0 \rangle$")
plt.plot(t,f_pol,label=r"$\vert \phi \rangle = \vert 000... \rangle$")
plt.plot(t,f_all_ones,label=r"$\vert \phi \rangle = \vert 111...10, k=0 \rangle$")
plt.title(r"$\lambda P^0 X P^0 + P^1 X P^0 + P^0 X P^1, \lambda=$"+str(V)+" , translationally invariant fidelities, N="+str(pxp.N))
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \phi \vert e^{-i H t} \vert Z_2,k=0\rangle \vert^2$")
plt.legend()
plt.show()
    
