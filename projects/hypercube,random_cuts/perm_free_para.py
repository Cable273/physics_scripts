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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def krylov(v,H,krylov_size):
    #generate subspace by repeated application of H
    temp = v
    for n in range(0,krylov_size):
        temp = np.dot(H,temp)
        v = np.vstack((v,temp))

    #orthonormalize basis using QR decomposition
    v = np.transpose(v)
    #orthornormalize with qr decomp
    x,r = np.linalg.qr(v)
    return x

def find_hamming_sectors(state_bits,system):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,system.N+1):
        hamming_sectors[n] = []
    for n in range(0,system.dim):
        h = 0
        for m in range(0,system.N,1):
            if system.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],system.basis_refs[n])
    return hamming_sectors

def sublattice_state(v1,v2,system):
    state_bits = np.zeros(system.N)
    for n in range(0,np.size(v1)):
        state_bits[2*v1[n]] = 1
    for n in range(0,np.size(v2)):
        state_bits[2*v2[n]+1] = 1
    return bin_state(state_bits,system)

def perm_key(v):
    return bin_to_int_base_m(v,int(pxp.N/2)+1)


#pxp cube
N=10
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H=spin_Hamiltonian(pxp,"x")
H.gen()

#init
sector_refs = dict()
for n in range(0,int(pxp.N/2)+1):
    for m in range(0,int(pxp.N/2)+1):
        sector_refs[perm_key([n,m])] = []

from_sector = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    c1=0
    c2=0
    for m in range(0,pxp.N):
        if pxp.basis[n,m] !=0:
            if m % 2 == 0:
                c1 = c1+1
            else:
                c2 = c2+1
    from_sector[pxp.basis_refs[n]] = np.array([c1,c2])
    sector_refs[perm_key(from_sector[pxp.basis_refs[n]])] = np.append(sector_refs[perm_key(from_sector[pxp.basis_refs[n]])],pxp.basis_refs[n])


#perm approx for comp, FSA
perm_basis = np.zeros(pxp.dim)
for n in range(0,len(sector_refs)):
    if np.size(sector_refs[n])>1e-5:
        temp_state = np.zeros(pxp.dim)
        for m in range(0,np.size(sector_refs[n],axis=0)):
            temp_state[pxp.keys[sector_refs[n][m]]] = 1
        temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
        perm_basis = np.vstack((perm_basis,temp_state))
perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))
perm_basis = np.unique(perm_basis,axis=1)

H.sector.find_eig()
print(H.sector.eigvalues())

H_perm = np.dot(np.transpose(np.conj(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
e,u = np.linalg.eigh(H_perm)
gs = u[:,0]
gs_comp = np.dot(perm_basis,gs)
gs_full = H.sector.eigvectors()[:,0]
print(np.abs(np.vdot(gs_comp,gs_full))**2)

