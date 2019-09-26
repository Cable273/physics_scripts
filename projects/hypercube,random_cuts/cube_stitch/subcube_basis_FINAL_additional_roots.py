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
N=12
pxp = unlocking_System([0],"periodic",2,N)
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

#finding mapping to and from sectors
to_sectors = dict()
for n in range(0,pxp.dim):
    non_zero_mapping = []
    for m in range(0,np.size(H.sector.matrix()[:,n],axis=0)):
        if np.abs(H.sector.matrix()[m,n])>1e-5:
            non_zero_mapping = np.append(non_zero_mapping,pxp.basis_refs[m])
    maps_to = np.zeros((np.size(non_zero_mapping),2))
    for m in range(0,np.size(maps_to,axis=0)):
        maps_to[m] = from_sector[int(non_zero_mapping[m])]
    to_sectors[pxp.basis_refs[n]] = np.unique(maps_to,axis=0)

def power(H,n,e,u):
    diag = np.power(e,n)
    return np.dot(u,np.dot(np.diag(diag),np.conj(np.transpose(u))))


#find roots from root sector to target sector
def find_root_refs(root_sector,target_sector):
    no_actions = np.sum(target_sector-root_sector)
    if no_actions > 1:
        H_power = power(H.sector.matrix(),no_actions,H.sector.eigvalues(),H.sector.eigvectors())
    else:
        H_power = H.sector.matrix()

    root_refs = sector_refs[perm_key(root_sector)]
    roots_kept = []
    for m in range(0,np.size(root_refs,axis=0)):
        maps_to = []
        for u in range(pxp.dim):
            if np.abs(H_power[pxp.keys[root_refs[m]],u])>1e-5:
                maps_to = np.append(maps_to,pxp.basis_refs[u])
        for u in range(0,np.size(maps_to,axis=0)):
            # print(from_sector[maps_to[u]],target_sector)
            if (from_sector[maps_to[u]] == target_sector).all():
                roots_kept = np.append(roots_kept,root_refs[m])
                break
    return roots_kept

def find_subcube_basis(root_ref,bits_to_add,LR):
    root_bits = pxp.basis[pxp.keys[root_ref]]
    one_loc = []
    for m in range(0,np.size(root_bits,axis=0)):
        if np.abs(root_bits[m])>1e-5:
            one_loc = np.append(one_loc,m)
    poss_zero_loc = []
    for m in range(0,np.size(root_bits,axis=0)):
        if m == 0:
            mm1 = pxp.N-1
        else:
            mm1 = m - 1

        if m == pxp.N-1:
            mp1 = 0
        else:
            mp1 = m + 1

        if LR == "Right":
            if m % 2 != 0:
                if root_bits[mm1] == 0 and root_bits[mp1] == 0 and root_bits[m] == 0:
                    poss_zero_loc = np.append(poss_zero_loc,m)

        if LR == "Left":
            if m % 2 == 0:
                if root_bits[mm1] == 0 and root_bits[mp1] == 0 and root_bits[m] == 0:
                    poss_zero_loc = np.append(poss_zero_loc,m)

    bit_loc = np.append(one_loc,poss_zero_loc)
    dim = np.sum(from_sector[root_ref])+bits_to_add
    subcube_system = unlocking_System([0,1],"open",2,dim)
    subcube_system.gen_basis()
    init_bits = np.append(np.ones(np.size(one_loc)),np.zeros(np.size(poss_zero_loc)))
    subcube_hamming = find_hamming_sectors(init_bits,subcube_system)
    basis = np.zeros((pxp.dim,len(subcube_hamming)))
    if np.size(poss_zero_loc) == bits_to_add:
        for m in range(0,len(subcube_hamming)):
            temp_state = np.zeros(pxp.dim)
            refs = subcube_hamming[m]
            for u in range(0,np.size(refs,axis=0)):
                sub_bits = subcube_system.basis[subcube_system.keys[refs[u]]]
                temp_bits = np.zeros(pxp.N)
                for k in range(0,np.size(sub_bits,axis=0)):
                    temp_bits[int(bit_loc[k])] = sub_bits[k]
                temp_ref = bin_to_int_base_m(temp_bits,pxp.base)
                temp_state[pxp.keys[temp_ref]] = 1
            temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
            basis[:,m] = temp_state
        return basis
    else:
        return basis

H.sector.find_eig()

subcube_rootsL = dict()
sector_labelsL = np.arange(pxp.N/2-2,1,-1)
root_sectorsL = np.zeros((np.size(sector_labelsL),2))
for n in range(0,np.size(sector_labelsL,axis=0)):
    root_sectorsL[n] = np.array([sector_labelsL[n],n])
print(root_sectorsL)
# root_sectorsL = np.vstack((root_sectorsL,np.array([2,1])))
# root_sectorsL = np.delete(root_sectorsL,np.size(root_sectorsL,axis=0)-1,axis=0)

subcube_rootsR = dict()
sector_labelsR = np.arange(pxp.N/2-2,1,-1)
root_sectorsR = np.zeros((np.size(sector_labelsR),2))
for n in range(0,np.size(sector_labelsR,axis=0)):
    root_sectorsR[n] = np.array([n,sector_labelsL[n]])
# root_sectorsR = np.vstack((root_sectorsR,np.array([1,2])))
# root_sectorsR = np.delete(root_sectorsR,np.size(root_sectorsR,axis=0)-1,axis=0)

roots = find_root_refs(np.array([3,0]),np.array([3,1]))
basis = find_subcube_basis(roots[0],1,"Right")
for m in range(1,np.size(roots,axis=0)):
    basis = basis + find_subcube_basis(roots[m],1,"Right")
for m in range(0,np.size(basis,axis=1)):
    basis[:,m] = basis[:,m] / np.power(np.vdot(basis[:,m],basis[:,m]),0.5)

roots = find_root_refs(np.array([2,0]),np.array([2,1]))
temp_basis = find_subcube_basis(roots[0],1,"Right")
for m in range(1,np.size(roots,axis=0)):
    temp_basis = temp_basis + find_subcube_basis(roots[m],1,"Right")
for m in range(0,np.size(temp_basis,axis=1)):
    temp_basis[:,m] = temp_basis[:,m] / np.power(np.vdot(basis[:,m],basis[:,m]),0.5)
basis = np.hstack((basis,temp_basis))

roots = find_root_refs(np.array([0,2]),np.array([1,2]))
temp_basis = find_subcube_basis(roots[0],1,"Left")
for m in range(1,np.size(roots,axis=0)):
    temp_basis = temp_basis + find_subcube_basis(roots[m],1,"Left")
for m in range(0,np.size(temp_basis,axis=1)):
    temp_basis[:,m] = temp_basis[:,m] / np.power(np.vdot(basis[:,m],basis[:,m]),0.5)
basis = np.hstack((basis,temp_basis))

roots = find_root_refs(np.array([0,3]),np.array([1,3]))
temp_basis = find_subcube_basis(roots[0],1,"Left")
for m in range(1,np.size(roots,axis=0)):
    temp_basis = temp_basis + find_subcube_basis(roots[m],1,"Left")
for m in range(0,np.size(temp_basis,axis=1)):
    temp_basis[:,m] = temp_basis[:,m] / np.power(np.vdot(basis[:,m],basis[:,m]),0.5)
basis = np.hstack((basis,temp_basis))

roots = find_root_refs(np.array([2,0]),np.array([2,2]))
temp_basis = find_subcube_basis(roots[0],2,"Right")
for m in range(1,np.size(roots,axis=0)):
    temp_basis = temp_basis + find_subcube_basis(roots[m],2,"Right")
for m in range(0,np.size(temp_basis,axis=1)):
    temp_basis[:,m] = temp_basis[:,m] / np.power(np.vdot(basis[:,m],basis[:,m]),0.5)
basis = np.hstack((basis,temp_basis))

neel_subcube_system = unlocking_System([0,1],"open",2,int(pxp.N/2))
neel_subcube_system.gen_basis()
z=ref_state(np.max(neel_subcube_system.basis_refs),neel_subcube_system)
neel_hamming = find_hamming_sectors(z.bits,neel_subcube_system)

#hypercube from Neel/AntiNeel
z0=zm_state(2,1,pxp)
z1=zm_state(2,1,pxp,1)
cube_dim = int(pxp.N/2)
cube_basisL = np.zeros(pxp.dim)
cube_basisR = np.zeros(pxp.dim)
for n in range(0,len(neel_hamming)):
# for n in range(0,2):
    refs = neel_hamming[n]
    temp_stateL = np.zeros(pxp.dim)
    temp_stateR = np.zeros(pxp.dim)
    one_locL = np.arange(0,pxp.N-1,2)
    one_locR = np.arange(1,pxp.N,2)
    for m in range(0,np.size(refs,axis=0)):
        state_bits = neel_subcube_system.basis[neel_subcube_system.keys[refs[m]]]
        temp_bitsL = np.zeros(pxp.N)
        temp_bitsR = np.zeros(pxp.N)
        for i in range(0,np.size(state_bits,axis=0)):
            temp_bitsL[int(one_locL[i])] = state_bits[i]
            temp_bitsR[int(one_locR[i])] = state_bits[i]
        temp_refL = bin_to_int_base_m(temp_bitsL,pxp.base)
        temp_refR = bin_to_int_base_m(temp_bitsR,pxp.base)
        temp_stateL = temp_stateL + ref_state(temp_refL,pxp).prod_basis()
        temp_stateR = temp_stateR + ref_state(temp_refR,pxp).prod_basis()
    temp_stateL = temp_stateL / np.power(np.vdot(temp_stateL,temp_stateL),0.5)
    temp_stateR = temp_stateR / np.power(np.vdot(temp_stateR,temp_stateR),0.5)
    # cube_basisL[:,n] = temp_stateL
    # cube_basisR[:,n] = temp_stateR
    cube_basisL = np.vstack((cube_basisL,temp_stateL))
    cube_basisR = np.vstack((cube_basisR,temp_stateR))
cube_basisL = np.transpose(np.delete(cube_basisL,0,axis=0))
cube_basisR = np.transpose(np.delete(cube_basisR,0,axis=0))

#combine total subsector basis
basis = np.hstack((cube_basisL,basis))
basis = np.hstack((basis,cube_basisR))

# project Hamiltonian + dynamics
basis = np.unique(basis,axis=1)
basis,temp = np.linalg.qr(basis)
from Diagnostics import print_wf
for n in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,n],pxp,1e-2)
    

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
e,u = np.linalg.eigh(H_rot)
overlap = np.log10(np.abs(u[1,:])**2)

H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
plt.scatter(e,overlap,marker="s",s=200,alpha=0.6,color="red",label="subcube")

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

H_perm = np.dot(np.transpose(np.conj(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
ep,up = np.linalg.eigh(H_perm)
overlap_perm = np.log10(np.abs(up[0,:])**2)
plt.scatter(ep,overlap_perm,marker="x",s=100,label="Perm")

#fsa
z=zm_state(2,1,pxp)
fsa_hamming = find_hamming_sectors(z.bits,pxp)
fsa_basis = np.zeros((pxp.dim,len(fsa_hamming)))
for n in range(0,len(fsa_hamming)):
    temp_state = np.zeros(pxp.dim)
    refs = fsa_hamming[n]
    for m in range(0,np.size(refs,axis=0)):
        temp_state[pxp.keys[refs[m]]] = 1
    temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    fsa_basis[:,n] = temp_state

H_fsa = np.dot(np.transpose(np.conj(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
ef,uf = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(uf[0,:])**2)
plt.scatter(ef,overlap_fsa,marker="D",s=100,color="green",alpha =0.6,label="FSA")
plt.legend()
plt.show()

u_comp_basis = np.dot(basis,u)
u_comp_basis_perm = np.dot(perm_basis,up)
u_comp_basis_fsa = np.dot(fsa_basis,uf)

exact_overlap = np.zeros(np.size(u_comp_basis,axis=1))
for n in range(0,np.size(u_comp_basis,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap

exact_overlap_perm = np.zeros(np.size(u_comp_basis_perm,axis=1))
for n in range(0,np.size(u_comp_basis_perm,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis_perm[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_perm[n] = max_overlap

exact_overlap_fsa = np.zeros(np.size(u_comp_basis_fsa,axis=1))
for n in range(0,np.size(u_comp_basis_fsa,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis_fsa[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_fsa[n] = max_overlap


plt.scatter(e,exact_overlap,marker="s",color="red",s=100,alpha=0.6,label="Subcube")
plt.scatter(ep,exact_overlap_perm,marker="x",s=100,label="Perm")
plt.scatter(ef,exact_overlap_fsa,label="FSA",marker="D",s=100,color="green")
plt.legend()
plt.show()

plt.matshow(np.abs(H_rot))
plt.show()
plt.matshow(np.abs(H_perm))
plt.show()
plt.matshow(np.abs(H_fsa))
plt.show()

print("\n")
print("subcube dim="+str(np.size(basis,axis=1)))
print("perm dim="+str(np.size(perm_basis,axis=1)))
print("fsa dim="+str(np.size(fsa_basis,axis=1)))
 

