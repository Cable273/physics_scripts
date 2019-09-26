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

def find_hamming_sectors(state_bits):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,pxp.N+1):
        hamming_sectors[n] = []
    for n in range(0,pxp.dim):
        h = 0
        for m in range(0,pxp.N,1):
            if pxp.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],pxp.basis_refs[n])
    return hamming_sectors

#init small hypercube
N=12
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
to_remove_refs = []
for n in range(int(pxp.N/2)+1,len(hamming_sectors)):
    to_remove_refs = np.append(to_remove_refs,hamming_sectors[n])

for n in range(0,np.size(to_remove_refs,axis=0)):
    if np.abs(to_remove_refs[n] )<1e-10:
        print("ZERO DELETED")
#redo basis
pxp.basis_refs_new = np.zeros(np.size(pxp.basis_refs)-np.size(to_remove_refs))
c=0
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    if pxp.basis_refs[n] not in to_remove_refs:
        pxp.basis_refs_new[c] = pxp.basis_refs[n]
        c = c+1
pxp.basis_refs = pxp.basis_refs_new

pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
for n in range(0,np.size(pxp.basis_refs)):
    pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
pxp.keys = dict()
for n in range(0,np.size(pxp.basis_refs)):
    pxp.keys[int(pxp.basis_refs[n])] = n
pxp.dim = np.size(pxp.basis_refs)

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
hamming_length = 0
for n in range(0,len(hamming_sectors)):
    if np.size(hamming_sectors[n])!=0:
        hamming_length = hamming_length + 1

H=spin_Hamiltonian(pxp,"x")
H.gen()

plt.matshow(np.abs(H.sector.matrix()))
plt.show()

def gaussian(x,mu,sigma):
    return 1/np.power(2*math.pi*np.power(sigma,2),0.5)*np.exp(-(x-mu)**2/(2*np.power(sigma,2)))

x=np.arange(0,2*(hamming_length)-1)
std = 1.5
fraction=2
k_cube = 2
k_int = 1
N_max_cube = 2

dist=gaussian(x,pxp.N/2,std)
print(dist)
x=x[0:int(np.size(x)/2)]
dist=dist[0:int(np.size(dist)/2)]
#rescale for
middle_hamming_size = np.size(hamming_sectors[hamming_length-1])
dist = dist/np.max(dist)*middle_hamming_size*fraction
dist = np.flip(np.round(dist)).astype(int)

print(dist)
dim_total = int(pxp.dim+np.sum(dist))
H_total = np.zeros((dim_total,dim_total))
H_total[0:pxp.dim,0:pxp.dim] = H.sector.matrix()
base_index = pxp.dim+1

root_index = base_index
for n in range(0,np.size(dist)-1):
    block_size = dist[n]
    next_block = np.arange(0,dist[n+1])
    if np.size(next_block)>0:
        a = np.random.uniform(0,1)
        if a < k_int:
            for m in range(0,block_size):
                coupling_ref = np.random.choice(next_block)
                H_total[root_index+m,root_index+block_size+coupling_ref-1]=1
                H_total[root_index+block_size+coupling_ref-1,root_index+m]=1
    root_index = root_index + dist[n]
d_extra = np.size(H_total,axis=0)-pxp.dim
D = np.size(H_total,axis=0)
H_total[pxp.dim:D,pxp.dim:D] = H_total[pxp.dim:D,pxp.dim:D]+ np.diag(np.ones(d_extra-1),1)+np.diag(np.ones(d_extra-1),-1)
# H_total[pxp.dim:D,pxp.dim:D] =  np.diag(np.ones(d_extra-1),1)+np.diag(np.ones(d_extra-1),-1)


#couple extra states to neighbouring columns (Hamming)
#0 first as only backwards coupling
base_index = pxp.dim
c=0
for n in range(0,dist[0]):
    for j in range(0,np.random.choice(np.arange(0,N_max_cube))):
        a = np.random.uniform(0,1)
        if a < k_cube:
            ref = np.random.choice(hamming_sectors[hamming_length-2])
            ref_index = pxp.keys[ref]
            H_total[base_index+c,ref_index] = 1
            H_total[ref_index,base_index+c] = 1
    c=c+1

for n in range(1,np.size(dist,axis=0)-2):
    for m in range(0,dist[n]):
        for j in range(0,np.random.choice(np.arange(0,N_max_cube))):
            a = np.random.uniform(0,1)
            if a < k_cube:
                ref_forward = np.random.choice(hamming_sectors[hamming_length-1-n-1])
                ref_backward = np.random.choice(hamming_sectors[hamming_length-1-n+1])
                ref_forward_index = pxp.keys[ref_forward]
                ref_backward_index = pxp.keys[ref_backward]

                H_total[base_index+c,ref_forward_index] = 1
                H_total[ref_forward_index,base_index+c] = 1

                H_total[base_index+c,ref_backward_index] = 1
                H_total[ref_backward_index,base_index+c] = 1
        c=c+1

for n in range(0,dist[np.size(dist,axis=0)-2]):
    for j in range(0,np.random.choice(np.arange(0,N_max_cube))):
        a = np.random.uniform(0,1)
        if a < k_cube:
            ref = np.random.choice(hamming_sectors[1])
            ref_index = pxp.keys[ref]
            H_total[base_index+c,ref_index] = 1
            H_total[ref_index,base_index+c] = 1
    c=c+1
    
plt.matshow(np.abs(H_total))
plt.show()

e,u = np.linalg.eigh(H_total)
z=zm_state(2,1,pxp)
z_energy = np.conj(u[pxp.keys[z.ref],:])
t=np.arange(0,10,0.1)
f=np.zeros(np.size(t))
for n in range(0,np.size(f,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.show()
pbar=ProgressBar()

for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
# for n in pbar(range(0,np.size(e,axis=0))):
    z_energy = np.conj(u[n,:])
    f=np.zeros(np.size(t))
    for m in range(0,np.size(f,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[m])
        f[m] = np.abs(np.vdot(evolved_state,z_energy))**2

    for m in range(0,np.size(f,axis=0)):
        if f[m] <0.2:
            cut = m
            break
    f_max = np.max(f[cut:])
    plt.plot(t,f)
plt.title(r"$\textrm{Half Hypercube, Cycle Disorder}, \sigma = 1.5, f=2, R_{cube} = 1, R_{int} = 1, k = 2, N=12$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2 $")
plt.show()

z=zm_state(2,1,pxp)
z_energy = np.conj(u[pxp.keys[z.ref],:])
overlap = np.log10(np.abs(z_energy)**2)
eigenvalues = np.copy(e)
to_del=[]
for m in range(0,np.size(overlap,axis=0)):
    if overlap[m] <-5:
        to_del = np.append(to_del,m)
for m in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[m])
    eigenvalues=np.delete(eigenvalues,to_del[m])
plt.scatter(eigenvalues,overlap)
plt.title(r"$\textrm{Half Hypercube, Cycle Disorder}, \sigma = 1.5, f=2, R_{cube} = 1, R_{int} = 1, k = 2, N=12$")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log \vert \langle E \vert \psi \rangle \vert^2$")
plt.show()
plt.show()

r_diff = np.zeros(np.size(e)-1)
for n in range(0,np.size(r_diff,axis=0)):
    r_diff[n] = np.abs(e[n+1]-e[n])

level_ratios = np.zeros(np.size(r_diff)-1)
for n in range(0,np.size(level_ratios,axis=0)):
    level_ratios[n] = np.min(np.array((r_diff[n],r_diff[n+1])))/np.max(np.array((r_diff[n],r_diff[n+1])))
mean_level_ratios = np.sum(level_ratios)/np.size(level_ratios)
print(mean_level_ratios)
