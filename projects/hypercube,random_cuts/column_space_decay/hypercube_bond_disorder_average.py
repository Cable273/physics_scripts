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

trial = int(sys.argv[1])

#init small hypercube
N=12
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()

# z=zm_state(2,1,pxp)
# hamming_sectors = find_hamming_sectors(z.bits)
# to_remove_refs = []
# for n in range(int(pxp.N/2)+1,len(hamming_sectors)):
    # to_remove_refs = np.append(to_remove_refs,hamming_sectors[n])

# for n in range(0,np.size(to_remove_refs,axis=0)):
    # if np.abs(to_remove_refs[n] )<1e-10:
        # print("ZERO DELETED")
# #redo basis
# pxp.basis_refs_new = np.zeros(np.size(pxp.basis_refs)-np.size(to_remove_refs))
# c=0
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # if pxp.basis_refs[n] not in to_remove_refs:
        # pxp.basis_refs_new[c] = pxp.basis_refs[n]
        # c = c+1
# pxp.basis_refs = pxp.basis_refs_new

# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
# pxp.keys = dict()
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.keys[int(pxp.basis_refs[n])] = n
# pxp.dim = np.size(pxp.basis_refs)

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
hamming_length = 0
for n in range(0,len(hamming_sectors)):
    if np.size(hamming_sectors[n])!=0:
        hamming_length = hamming_length + 1

H=spin_Hamiltonian(pxp,"x")
H.gen()

# plt.matshow(np.abs(H.sector.matrix()))
# plt.show()

def gaussian(x,mu,sigma):
    return 1/np.power(2*math.pi*np.power(sigma,2),0.5)*np.exp(-(x-mu)**2/(2*np.power(sigma,2)))

# x=np.arange(0,2*hamming_length-1)
x=np.arange(0,hamming_length)
std = 0.8
fraction=0.1
k=2

# dist=gaussian(x,pxp.N/2,std)
# x=x[0:int(np.size(x)/2)]
# dist=dist[0:int(np.size(dist)/2)]
# #rescale for
# middle_hamming_size = np.size(hamming_sectors[hamming_length-1])
# dist = dist/np.max(dist)*middle_hamming_size*fraction
# dist = np.round(dist).astype(int)

dist=gaussian(x,pxp.N/2,std)
# plt.plot(dist,marker="s")
# plt.show()
#rescale for
middle_hamming_size = np.size(hamming_sectors[hamming_length-1])
middle_hamming_size = np.size(hamming_sectors[(hamming_length-1)/2])
dist = dist/np.max(dist)*middle_hamming_size*fraction
dist = np.round(dist).astype(int)

H_total = H.sector.matrix()
# for n in range(1,np.size(dist)-1):
for n in range(1,int(np.size(dist,axis=0)/2)):
    if dist[n] !=0:
        refs = np.random.choice(hamming_sectors[n],dist[n])
        forward_refs = dict()
        for m in range(0,k):
            forward_refs[m] = np.random.choice(hamming_sectors[n+1],dist[n])
        for m in range(0,np.size(refs,axis=0)):
            for j in range(0,len(forward_refs)):
                H_total[pxp.keys[refs[m]],pxp.keys[forward_refs[j][m]]] = 1
                H_total[pxp.keys[forward_refs[j][m]],pxp.keys[refs[m]]] = 1

for n in range(np.size(dist,axis=0)-2,int(np.size(dist,axis=0)/2),-1):
    if dist[n] !=0:
        refs = np.random.choice(hamming_sectors[n],dist[n])
        forward_refs = dict()
        for m in range(0,k):
            forward_refs[m] = np.random.choice(hamming_sectors[n-1],dist[n])
        for m in range(0,np.size(refs,axis=0)):
            for j in range(0,len(forward_refs)):
                H_total[pxp.keys[refs[m]],pxp.keys[forward_refs[j][m]]] = 1
                H_total[pxp.keys[forward_refs[j][m]],pxp.keys[refs[m]]] = 1

middle_index = int(np.size(dist)/2)
# middle_index = np.size(dist)-1
if dist[middle_index] !=0:
    refs = np.random.choice(hamming_sectors[middle_index],dist[middle_index])
    forward_refs = dict()
    backward_refs = dict()
    for m in range(0,k):
        forward_refs[m] = np.random.choice(hamming_sectors[middle_index+1],dist[middle_index])
        backward_refs[m] = np.random.choice(hamming_sectors[middle_index-1],dist[middle_index])
    for m in range(0,np.size(refs,axis=0)):
        for j in range(0,len(forward_refs)):
            H_total[pxp.keys[refs[m]],pxp.keys[forward_refs[j][m]]] = 1
            H_total[pxp.keys[forward_refs[j][m]],pxp.keys[refs[m]]] = 1

            H_total[pxp.keys[refs[m]],pxp.keys[backward_refs[j][m]]] = 1
            H_total[pxp.keys[backward_refs[j][m]],pxp.keys[refs[m]]] = 1

np.save("./disorder_realizations/bond_disorder/H,"+str(trial),H_total)
# plt.matshow(np.abs(H_total))
# plt.show()
# e,u = np.linalg.eigh(H_total)

# z=zm_state(2,1,pxp)
# # z=ref_state(0,pxp)
# z_energy = u[pxp.keys[z.ref],:]
# t=np.arange(0,10,0.1)
# f=np.zeros(np.size(t))
# for n in range(0,np.size(f,axis=0)):
    # evolved_state = time_evolve_state(z_energy,e,t[n])
    # f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
# plt.plot(t,f)
# plt.show()

# pbar=ProgressBar()
# for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    # z=ref_state(pxp.basis_refs[n],pxp)
    # z_energy = u[pxp.keys[z.ref],:]
    # f=np.zeros(np.size(t))
    # for m in range(0,np.size(f,axis=0)):
        # evolved_state = time_evolve_state(z_energy,e,t[m])
        # f[m] = np.abs(np.vdot(evolved_state,z_energy))**2
    # plt.plot(t,f)
# plt.title(r"$\textrm{Hypercube, Bond Disorder, Quench Fidelity}, N=12$")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2 $")
# plt.show()
# plt.show()

# z=zm_state(2,1,pxp)
# z_energy = u[pxp.keys[z.ref],:]
# overlap =np.log10(np.abs(z_energy)**2)
# to_del=[]
# eigvalues = np.copy(e)
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-5:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigvalues=np.delete(eigvalues,to_del[n])
    
# plt.title(r"$\textrm{Hypercube, Bond Disorder, Eigenstate overlap with Neel}, N=12$")
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log \vert \langle E \vert \psi \rangle \vert^2$")
# plt.scatter(eigvalues,overlap)
# plt.show()

# ent = entropy(pxp)
# ent_vals = np.zeros(np.size(e))
# for n in range(0,np.size(ent_vals,axis=0)):
    # ent_vals[n] = ent.eval(u[:,n])
# plt.title(r"$\textrm{Hypercube, Bond Disorder, Eigenstate Entropy}, N=12$")
# plt.xlabel(r"$E$")
# plt.ylabel(r"$S$")
# plt.scatter(e,ent_vals)
# plt.show()
