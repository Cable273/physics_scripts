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

#pxp cube
N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])


#smaller hypercubes + hamming, for subcube identification
sub_cube_systems = dict()
sub_cube_hamming = dict()
for n in range(int(pxp.N/2),1,-1):
    sub_cube_systems[n] = unlocking_System([0,1],"open",2,n)
    sub_cube_systems[n].gen_basis()
    z_temp = np.ones(sub_cube_systems[n].N)
    z_temp[np.size(z_temp,axis=0)-1] = 0
    z_temp = bin_state(z_temp,sub_cube_systems[n])
    sub_cube_hamming[n] = find_hamming_sectors(z_temp.bits,sub_cube_systems[n])

z=zm_state(2,1,pxp)
full_hamming = find_hamming_sectors(z.bits,pxp)

root_statesL = dict()
root_statesR = dict()
root_statesL[0] = sublattice_state(np.array([1,2]),[],pxp)
root_statesL[1] = sublattice_state(np.array([2,3]),[],pxp)
root_statesL[2] = sublattice_state(np.array([3,0]),[],pxp)
root_statesL[3] = sublattice_state(np.array([0,1]),[],pxp)

root_statesR[0] = sublattice_state(np.array([]),[1,2],pxp)
root_statesR[1] = sublattice_state(np.array([]),[2,3],pxp)
root_statesR[2] = sublattice_state(np.array([]),[3,0],pxp)
root_statesR[3] = sublattice_state(np.array([]),[0,1],pxp)

#generate subcube basis states, using smaller cube basis + root node
subcube_root_hamming_refsL = dict()
subcube_root_hamming_refsR = dict()
for n in range(0,len(root_statesL)):
    subcube_root_hamming_refsL[n] = dict()
    subcube_root_hamming_refsR[n] = dict()
    #index of 1s
    one_locL = []
    zero_locL = []
    one_locR = []
    zero_locR = []
    for m in range(0,np.size(root_statesL[n].bits,axis=0)):
        if root_statesL[n].bits[m] == 1:
            one_locL = np.append(one_locL,m)

        if root_statesR[n].bits[m] == 1:
            one_locR = np.append(one_locR,m)

        if m % 2 != 0:
            if m == 0:
                mm1 = np.size(root_statesL[n].bits)-1
            else:
                mm1 = m-1
            if m == np.size(root_statesL[n].bits)-1:
                mp1 = 0
            else:
                mp1 = m+1

            if root_statesL[n].bits[mm1] == 0 and root_statesL[n].bits[mp1] == 0 and root_statesL[n].bits[m] == 0:
                zero_locL = np.append(zero_locL,m)

        else:
            if m == 0:
                mm1 = np.size(root_statesL[n].bits)-1
            else:
                mm1 = m-1
            if m == np.size(root_statesL[n].bits)-1:
                mp1 = 0
            else:
                mp1 = m+1

            if root_statesR[n].bits[mm1] == 0 and root_statesR[n].bits[mp1] == 0 and root_statesR[n].bits[m] == 0:
                zero_locR = np.append(zero_locR,m)

    #hash table for generating basis states from subcube basis
    sub_cube_dim = int(np.size(one_locL)+np.size(zero_locL))
    index_redirectL = np.zeros(sub_cube_dim)
    index_redirectR = np.zeros(sub_cube_dim)
    c=0
    for m in range(0,np.size(one_locL,axis=0)):
        index_redirectL[c]=one_locL[m]
        index_redirectR[c]=one_locR[m]
        c=c+1
    for m in range(0,np.size(zero_locL,axis=0)):
        index_redirectL[c] = zero_locL[m]
        index_redirectR[c] = zero_locR[m]
        c=c+1


    for m in range(0,len(sub_cube_hamming[sub_cube_dim])):
        subcube_root_hamming_refsL[n][m] = np.zeros(np.size(sub_cube_hamming[sub_cube_dim][m]))
        subcube_root_hamming_refsR[n][m] = np.zeros(np.size(sub_cube_hamming[sub_cube_dim][m]))
        for j in range(0,np.size(sub_cube_hamming[sub_cube_dim][m],axis=0)):

            state_bin = sub_cube_systems[sub_cube_dim].basis[sub_cube_systems[sub_cube_dim].keys[sub_cube_hamming[sub_cube_dim][m][j]]]
            temp_stateL = np.zeros(pxp.N)
            temp_stateR = np.zeros(pxp.N)
            for k in range(0,np.size(state_bin,axis=0)):
                temp_stateL[int(index_redirectL[k])] = state_bin[k]
                temp_stateR[int(index_redirectR[k])] = state_bin[k]
            temp_refL = bin_to_int_base_m(temp_stateL,pxp.base)
            temp_refR = bin_to_int_base_m(temp_stateR,pxp.base)

            subcube_root_hamming_refsL[n][m][j] = temp_refL
            subcube_root_hamming_refsR[n][m][j] = temp_refR

subcube_root_hamming_basisL = dict()
subcube_root_hamming_basisR = dict()
for n in range(0,len(subcube_root_hamming_refsL)):
    subcube_root_hamming_basisL[n] = np.zeros((pxp.dim,len(subcube_root_hamming_refsL[n])))
    subcube_root_hamming_basisR[n] = np.zeros((pxp.dim,len(subcube_root_hamming_refsR[n])))
    for m in range(0,len(subcube_root_hamming_refsL[n])):
        tempL = np.zeros(pxp.dim)
        tempR = np.zeros(pxp.dim)
        for k in range(0,np.size(subcube_root_hamming_refsL[n][m],axis=0)):
            tempL = tempL + ref_state(subcube_root_hamming_refsL[n][m][k],pxp).prod_basis()
            tempR = tempR + ref_state(subcube_root_hamming_refsR[n][m][k],pxp).prod_basis()
        tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
        tempR = tempR / np.power(np.vdot(tempR,tempR),0.5)
        subcube_root_hamming_basisL[n][:,m] = tempL
        subcube_root_hamming_basisR[n][:,m] = tempR

subcube_combined_basisL = subcube_root_hamming_basisL[0]
subcube_combined_basisR = subcube_root_hamming_basisR[0]
for n in range(1,len(subcube_root_hamming_refsL)):
    subcube_combined_basisL = subcube_combined_basisL + subcube_root_hamming_basisL[n]
    subcube_combined_basisR = subcube_combined_basisR + subcube_root_hamming_basisR[n]

for n in range(0,np.size(subcube_combined_basisL,axis=1)):
    subcube_combined_basisL[:,n] = subcube_combined_basisL[:,n] / np.power(np.vdot(subcube_combined_basisL[:,n],subcube_combined_basisL[:,n]),0.5)
    subcube_combined_basisR[:,n] = subcube_combined_basisR[:,n] / np.power(np.vdot(subcube_combined_basisR[:,n],subcube_combined_basisR[:,n]),0.5)

sub_cube_dim = int(pxp.N/2)
z=ref_state(np.max(sub_cube_systems[sub_cube_dim].basis_refs),sub_cube_systems[sub_cube_dim])
hamming_sectors = find_hamming_sectors(z.bits,sub_cube_systems[sub_cube_dim])

half_cube_refsL = dict()
half_cube_refsR = dict()
for n in range(0,len(hamming_sectors)):
    half_cube_refsL[n] = np.zeros(np.size(hamming_sectors[n]))
    half_cube_refsR[n] = np.zeros(np.size(hamming_sectors[n]))
    one_locL = np.arange(0,pxp.N-1,2)
    one_locR = np.arange(1,pxp.N,2)
    for m in range(0,np.size(hamming_sectors[n],axis=0)):
        bits = sub_cube_systems[sub_cube_dim].basis[sub_cube_systems[sub_cube_dim].keys[hamming_sectors[n][m]]]
        tempL = np.zeros(pxp.N)
        tempR = np.zeros(pxp.N)
        for k in range(0,np.size(bits,axis=0)):
            tempL[one_locL[k]] = bits[k]
            tempR[one_locR[k]] = bits[k]
        refL = bin_to_int_base_m(tempL,pxp.base)
        refR = bin_to_int_base_m(tempR,pxp.base)

        half_cube_refsL[n][m] = refL
        half_cube_refsR[n][m] = refR

half_cube_basisL = np.zeros((pxp.dim,len(half_cube_refsL)))
half_cube_basisR = np.zeros((pxp.dim,len(half_cube_refsR)))

for n in range(0,len(half_cube_refsL)):
    tempL = np.zeros(pxp.dim)
    tempR = np.zeros(pxp.dim)
    for m in range(0,np.size(half_cube_refsL[n])):
        tempL = tempL + ref_state(half_cube_refsL[n][m],pxp).prod_basis()
        tempR = tempR + ref_state(half_cube_refsR[n][m],pxp).prod_basis()
    tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
    tempR = tempR / np.power(np.vdot(tempR,tempR),0.5)
    half_cube_basisL[:,n] = tempL
    half_cube_basisR[:,n] = tempR
        
#form total sub basis
basis = np.hstack((half_cube_basisL,subcube_combined_basisL))
basis = np.hstack((basis,subcube_combined_basisR))
basis = np.hstack((basis,half_cube_basisR))

from Diagnostics import print_wf
for n in range(0,np.size(basis,axis=1)):
    print("\n,State "+str(n))
    print_wf(basis[:,n],pxp,1e-5)

basis = np.unique(basis,axis=1)
basis,temp = np.linalg.qr(basis)

#FSA basis
z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits,pxp)
fsa_basis = np.zeros((pxp.dim,len(hamming_sectors)))
for n in range(0,len(hamming_sectors)):
    for m in range(0,np.size(hamming_sectors[n],axis=0)):
        fsa_basis[:,n] = fsa_basis[:,n] + ref_state(hamming_sectors[n][m],pxp).prod_basis()
    fsa_basis[:,n] = fsa_basis[:,n]/np.power(np.vdot(fsa_basis[:,n],fsa_basis[:,n]),0.5)

H = spin_Hamiltonian(pxp,"x") 
H.gen()
H.sector.find_eig()

krylov_basis = krylov(z.prod_basis(),H.sector.matrix(),pxp.N)

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
plt.matshow(np.abs(H_rot))
plt.show()

#subcube neel overlap
e,u = np.linalg.eigh(H_rot)
psi_energy = u[0,:]
overlap = np.log10(np.abs(psi_energy)**2)
plt.scatter(e,overlap,marker="x",color="red",s=200,label="Subcube basis")

#fsa neel overlap
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
psi_energy_fsa = u_fsa[0,:]
overlap_fsa = np.log10(np.abs(psi_energy_fsa)**2)
plt.scatter(e_fsa,overlap_fsa,marker="x",color="blue",s=200,label="FSA")

#krylov neel overlap
e_krylov,u_krylov = np.linalg.eigh(H_krylov)
psi_energy_krylov = u_krylov[0,:]
overlap_krylov = np.log10(np.abs(psi_energy_krylov)**2)
plt.scatter(e_krylov,overlap_krylov,marker="x",color="green",s=200,label="Krylov")

z=zm_state(2,1,pxp)
overlap = eig_overlap(z,H).eval()
plt.scatter(H.sector.eigvalues(),overlap,label="Exact")

plt.legend()
plt.show()

u_comp_basis = np.dot(basis,u)
exact_overlap = np.zeros(np.size(e),dtype=complex)
for n in range(0,np.size(e,axis=0)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.scatter(e,exact_overlap,label="Subcube")

u_comp_basis_fsa = np.dot(fsa_basis,u_fsa)
exact_overlap_fsa = np.zeros(np.size(e_fsa),dtype=complex)
for n in range(0,np.size(e_fsa,axis=0)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_comp_basis_fsa[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_fsa[n] = max_overlap
plt.scatter(e_fsa,exact_overlap_fsa,label="FSA")

u_comp_basis_krylov = np.dot(krylov_basis,u_krylov)
exact_overlap_krylov = np.zeros(np.size(e_krylov),dtype=complex)
for n in range(0,np.size(e_krylov,axis=0)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_comp_basis_krylov[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_krylov[n] = max_overlap
plt.scatter(e_krylov,exact_overlap_krylov,marker="x",s=200,label="Krylov")
plt.legend()
plt.show()
