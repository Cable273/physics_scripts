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

import numpy as np
import scipy as sp
import math

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

from subcube_functions import perm_key,cube_fsa,find_root_refs

N = 18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

sector_refs = dict()
from_sector = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    bits = pxp.basis[n]
    #find perm sector
    c1 = 0
    c2 = 0
    for m in range(0,np.size(bits,axis=0)):
        if bits[m] == 1:
            if m % 2 == 0:
                c1 = c1 + 1
            else:
                c2 = c2 + 1
    sector = np.array([c1,c2])
    from_sector[pxp.basis_refs[n]] = sector
    key = perm_key(sector,pxp)
    if key in sector_refs.keys():
        sector_refs[key] = np.append(sector_refs[key],pxp.basis_refs[n])
    else:
        sector_refs[key] = [pxp.basis_refs[n]]

keys = list(sector_refs.keys())
perm_basis = np.zeros(pxp.dim)
basis_labels = dict()
pbar=ProgressBar()
print("Generating perm basis")
for n in pbar(range(0,np.size(keys,axis=0))):
    temp_state = np.zeros(pxp.dim)
    refs = sector_refs[keys[n]]
    for m in range(0,np.size(refs,axis=0)):
        temp_state[pxp.keys[refs[m]]] = 1
    temp_state  = temp_state/np.power(np.vdot(temp_state,temp_state),0.5)
    perm_basis = np.vstack((perm_basis,temp_state))

    #use ref[0] to check sector
    in_sector = from_sector[refs[0]]
    basis_labels[n] = in_sector
perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))

def plot_perm_support(state,title):
    perm_support = np.zeros((int(pxp.N/2+1),int(pxp.N/2+1)),dtype=complex)
    for m in range(0,np.size(state,axis=0)):
        perm_support[basis_labels[m][0],basis_labels[m][1]] = state[m]
    plt.matshow(np.abs(perm_support))
    plt.colorbar()
    plt.title(str(title))
    plt.xlabel("n")
    plt.ylabel("m",rotation=0)
    plt.show()

H = spin_Hamiltonian(pxp,"x",pxp_syms)
e = np.load("./pxp,e,18.npy")
u = np.load("./pxp,u,18.npy")

z=zm_state(2,1,pxp)
overlap = np.log10(np.abs(u[pxp.keys[z.ref],:])**2)

#remove numerical zeros
e_kept = []
u_kept = np.zeros(pxp.dim)
overlap_kept = []
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] >= -5:
        e_kept = np.append(e_kept,e[n])
        u_kept = np.vstack((u_kept,u[:,n]))
        overlap_kept = np.append(overlap_kept,overlap[n])
u_kept = np.transpose(np.delete(u_kept,0,axis=0))
e = e_kept
u = u_kept
overlap = overlap_kept

def find_nearest_state(x0,y0,x_data,y_data):
    d = np.zeros(np.size(x_data))
    for n in range(0,np.size(d,axis=0)):
        d[n] = np.power((x_data[n]-x0)**2+(y_data[n]-y0)**2,0.5)
    min_index = np.argmin(d)

    # plt.scatter(e,overlap)
    # print("\n"+str(min_index)+", E="+str(x_data[min_index])+", Overlap="+str(y_data[min_index])+"\n")
    # plt.scatter(x_data[min_index],y_data[min_index],s=200,color="red",alpha=0.5)
    # plt.show()
    return min_index

#N=18, z2
scar_loc = []
scar_loc = np.append(scar_loc,find_nearest_state(1,-0.7,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(1,-1,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(1.8,-1,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(2.5,-1.6,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(2.9,-1.8,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(3.5,-2.25,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(4.2,-3,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(4.5,-3.5,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(5,-4.5,e,overlap))

scar_loc = np.append(scar_loc,find_nearest_state(-1,-0.7,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-1,-1,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-1.8,-1,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-2.5,-1.6,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-2.9,-1.8,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-3.5,-2.25,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-4.2,-3,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-4.5,-3.5,e,overlap))
scar_loc = np.append(scar_loc,find_nearest_state(-5,-4.5,e,overlap))
scar_loc = np.sort(scar_loc.astype(int))

subband_loc = []
subband_loc = np.append(subband_loc,find_nearest_state(4.1,-3.75,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(3.5,-3,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(3,-3,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(2.2,-2.2,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(2,-2.2,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(1.5,-1.5,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(0.3,-1.5,e,overlap))

subband_loc = np.append(subband_loc,find_nearest_state(-4.1,-3.75,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-3.5,-3,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-3,-3,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-2.2,-2.2,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-2,-2.2,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-1.5,-1.5,e,overlap))
subband_loc = np.append(subband_loc,find_nearest_state(-0.3,-1.5,e,overlap))
subband_loc = np.sort(subband_loc.astype(int))

#N=18,z3
# scar_loc = []
# scar_loc = np.append(scar_loc,find_nearest_state(1,-0.7,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(1,-1.2,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(1.5,-1.6,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(1.5,-1.9,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(2.4,-2.0,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(2.4,-2.3,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(3.2,-2.5,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(4.2,-3.5,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(4.2,-4.0,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(5.2,-5.0,e,overlap))

# scar_loc = np.append(scar_loc,find_nearest_state(-1,-0.7,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-1,-1.2,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-1.5,-1.6,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-1.5,-1.9,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-2.4,-2.0,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-2.4,-2.3,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-3.2,-2.5,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-4.2,-3.5,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-4.2,-4.0,e,overlap))
# scar_loc = np.append(scar_loc,find_nearest_state(-5.2,-5.0,e,overlap))
# scar_loc = np.sort(scar_loc.astype(int))


plt.scatter(e,overlap)
for n in range(0,np.size(scar_loc,axis=0)):
    plt.scatter(e[scar_loc[n]],overlap[scar_loc[n]],s=200,color="red",alpha=0.6)

for n in range(0,np.size(subband_loc,axis=0)):
    plt.scatter(e[subband_loc[n]],overlap[subband_loc[n]],s=200,color="cyan",alpha=0.6)
plt.show()
    
scar_weight_in_perm = np.zeros(np.size(scar_loc))
scar_e = np.zeros(np.size(scar_loc))
for n in range(0,np.size(scar_loc,axis=0)):
    scar_eig = u[:,int(scar_loc[n])]
    scar_eig_perm_basis = np.dot(np.conj(np.transpose(perm_basis)),scar_eig)
    scar_weight_in_perm[n] = np.abs(np.vdot(scar_eig_perm_basis,scar_eig_perm_basis))**2
    scar_e[n] = e[scar_loc[n]]
    plot_perm_support(scar_eig_perm_basis,title="Scarred Eigs, N=18")

plt.plot(scar_e,scar_weight_in_perm,marker="s")
plt.show()

subband_weight_in_perm = np.zeros(np.size(subband_loc))
subband_e = np.zeros(np.size(subband_loc))
for n in range(0,np.size(subband_loc,axis=0)):
    subband_eig = u[:,int(subband_loc[n])]
    subband_eig_perm_basis = np.dot(np.conj(np.transpose(perm_basis)),subband_eig)
    subband_weight_in_perm[n] = np.abs(np.vdot(subband_eig_perm_basis,subband_eig_perm_basis))**2
    subband_e[n] = e[subband_loc[n]]
    plot_perm_support(subband_eig_perm_basis,title="Subband Eigs, N=18")

plt.plot(subband_e,subband_weight_in_perm,marker="s")
plt.show()
