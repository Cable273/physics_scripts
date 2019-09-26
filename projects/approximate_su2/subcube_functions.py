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
import numpy as np
import scipy as sp
import math

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

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


import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def W(N):
    choose = ncr(int(N/2),2)
    return 1/np.power(choose,0.5)*1/np.power(int(N/2)-1,0.5)*N/2

def power(H,n,e,u):
    diag = np.power(e,n)
    return np.dot(u,np.dot(np.diag(diag),np.conj(np.transpose(u))))

#find roots from root sector to target sector
def find_root_refs(root_sector,target_sector,H,sector_refs,from_sector,system,exclude):
    #find excluded states in target sector
    exclude_sector = []
    for n in range(0,np.size(exclude,axis=0)):
        if (from_sector[exclude[n]] == target_sector).all():
            exclude_sector = np.append(exclude_sector,exclude[n])
    target_refs = sector_refs[perm_key(target_sector,system)]
    to_del=[]
    for n in range(0,np.size(target_refs,axis=0)):
        if target_refs[n] in exclude_sector:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        target_refs=np.delete(target_refs,to_del[n])
        
    no_actions = np.sum(target_sector-root_sector)
    if no_actions > 1:
        H_power = power(H.sector.matrix(),no_actions,H.sector.eigvalues(),H.sector.eigvectors())
    else:
        H_power = H.sector.matrix()

    root_refs = sector_refs[perm_key(root_sector,system)]
    roots_kept = []
    for m in range(0,np.size(root_refs,axis=0)):
        maps_to = []
        for u in range(0,system.dim):
            if np.abs(H_power[system.keys[root_refs[m]],u])>1e-5:
                maps_to = np.append(maps_to,system.basis_refs[u])

        for u in range(0,np.size(target_refs,axis=0)):
            if target_refs[u] in maps_to:
                roots_kept = np.append(roots_kept,root_refs[m])
                break
            
        # for u in range(0,np.size(maps_to,axis=0)):
            # if (from_sector[int(maps_to[u])] == target_sector).all():
                # roots_kept = np.append(roots_kept,root_refs[m])
                # break

    return roots_kept

def perm_key(sector,system):
    return bin_to_int_base_m(sector,int(system.N/2)+1)

def cube_fsa(root_sector,sublattice_parity,sector_refs,system):
    refs = sector_refs[perm_key(root_sector,system)]
    # #find root refs, those with two neighbouring 1->0 from Neel
    root_refs = []
    for n in range(0,np.size(refs,axis=0)):
        bits = system.basis[system.keys[refs[n]]]
        for m in range(0,np.size(bits,axis=0)):
            if m == np.size(bits)-1:
                mp1 = 0
                mp2 = 1
                mp3 = 2
            elif m == np.size(bits)-2:
                mp1 = m + 1
                mp2 = 0
                mp3 = 1
            elif m == np.size(bits)-3:
                mp1 = m + 1
                mp2 = m + 2
                mp3 = 0
            else:
                mp1 = m + 1
                mp2 = m + 2
                mp3 = m + 3

            if bits[m] == 0 and bits[mp1] == 0 and bits[mp2] == 0 and bits[mp3] == 0:
                root_refs = np.append(root_refs,refs[n])
                break

    root_bits = np.zeros((np.size(root_refs),system.N))
    for n in range(0,np.size(root_refs,axis=0)):
        root_bits[n] = system.basis[system.keys[root_refs[n]]]

    fsa_min_bit_loc = np.zeros((np.size(root_bits,axis=0),int(system.N/2)-2))
    fsa_plus_bit_loc = np.zeros(np.size(root_bits,axis=0))

    for n in range(0,np.size(root_bits,axis=0)):
        c=0
        for m in range(0,np.size(root_bits[n],axis=0)):
            if root_bits[n,m] == 1:
                fsa_min_bit_loc[n,c] = m
                c = c+1


            if sublattice_parity == "L":
                if m % 2 != 0:
                    if m == system.N-1:
                        mp1 = 0
                    else:
                        mp1 = m + 1
                    if m == 0:
                        mm1 = system.N-1
                    else:
                        mm1 = m - 1
                    if root_bits[n,mm1] == 0 and root_bits[n,m] == 0 and root_bits[n,mp1] == 0:
                        fsa_plus_bit_loc[n] = m

            elif sublattice_parity == "R":
                if m % 2 == 0:
                    if m == system.N-1:
                        mp1 = 0
                    else:
                        mp1 = m + 1
                    if m == 0:
                        mm1 = system.N-1
                    else:
                        mm1 = m - 1
                    if root_bits[n,mm1] == 0 and root_bits[n,m] == 0 and root_bits[n,mp1] == 0:
                        fsa_plus_bit_loc[n] = m

    fsa_plus = dict()
    fsa_min = dict()
    for n in range(0,np.size(fsa_plus_bit_loc,axis=0)):
        fsa_plus[n] = np.zeros((system.dim,system.dim))
        #scan basis + sites
        for m in range(0,np.size(system.basis_refs,axis=0)):
            for k in range(0,system.N):
                #sp
                if np.abs(k - fsa_plus_bit_loc[n])<1e-5:
                    bits = np.copy(system.basis[m])
                    if k == system.N-1:
                        kp1 = 0
                    else:
                        kp1 = k +1
                    if k == 0:
                        km1 = system.N-1
                    else:
                        km1 = k - 1

                    if bits[kp1] == 0 and bits[km1] == 0 and bits[k] == 0:
                        bits[k] = 1
                        new_ref = bin_to_int_base_m(bits,system.base)
                        fsa_plus[n][m,system.keys[new_ref]] = 1
                #sm
                if k in fsa_min_bit_loc[n]:
                    bits = np.copy(system.basis[m])
                    if k == system.N-1:
                        kp1 = 0
                    else:
                        kp1 = k +1
                    if k == 0:
                        km1 = system.N-1
                    else:
                        km1 = k - 1

                    if bits[kp1] == 0 and bits[km1] == 0 and bits[k] == 1:
                        bits[k] = 0
                        new_ref = bin_to_int_base_m(bits,system.base)
                        fsa_plus[n][m,system.keys[new_ref]] = 1

    for n in range(0,len(fsa_plus)):
        fsa_min[n] = np.conj(np.transpose(fsa_plus[n]))

    fsa_basis = dict()
    fsa_dim = int(system.N/2-2)
    for n in range(0,np.size(root_refs,axis=0)):
        fsa_basis[n] = ref_state(root_refs[n],system).prod_basis()
        current_state = fsa_basis[n]
        for m in range(0,fsa_dim):
            new_state = np.dot(fsa_min[n],current_state)
            new_state = new_state / np.power(np.vdot(new_state,new_state),0.5)
            fsa_basis[n] = np.vstack((fsa_basis[n],new_state))
            current_state = new_state
        fsa_basis[n] = np.transpose(fsa_basis[n])

    basis = fsa_basis[0]
    for n in range(1,len(fsa_basis)):
        basis = basis + fsa_basis[n]
    for n in range(0,np.size(basis,axis=1)):
        basis[:,n] = basis[:,n] / np.power(np.vdot(basis[:,n],basis[:,n]),0.5)
    return basis

