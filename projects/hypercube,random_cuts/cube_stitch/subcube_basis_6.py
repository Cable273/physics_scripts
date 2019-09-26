#!/usr/bin/env python# -*- coding: utf-8 -*-

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

#init small hypercube
N=6
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

pxp_sub = unlocking_System([0,1],"open",2,int(N/2))
pxp_sub.gen_basis()
z=ref_state(np.max(pxp_sub.basis_refs),pxp_sub)
hamming_sub = find_hamming_sectors(z.bits,pxp_sub)

c1_bits = dict()
c2_bits = dict()

for n in range(0,len(hamming_sub)-1):
    c1_bits[n] = np.zeros((np.size(hamming_sub[n]),pxp.N))
    c2_bits[n] = np.zeros((np.size(hamming_sub[n]),pxp.N))
    print("\n")
    for m in range(0,np.size(hamming_sub[n],axis=0)):
        bits = pxp_sub.basis[pxp_sub.keys[hamming_sub[n][m]]]
        for i in range(0,np.size(bits,axis=0)):
            c1_bits[n][m,2*i] = bits[i]
            c2_bits[n][m,2*i+1] = bits[i]
    print(c1_bits[n])
    print(c2_bits[n])

c1_refs = dict()
c2_refs = dict()
for n in range(0,len(c1_bits)):
    c1_refs[n] = np.zeros(np.size(c1_bits[n],axis=0))
    c2_refs[n] = np.zeros(np.size(c2_bits[n],axis=0))
    for m in range(0,np.size(c1_refs[n],axis=0)):
        c1_refs[n][m] = bin_to_int_base_m(c1_bits[n][m],pxp.base)
        c2_refs[n][m] = bin_to_int_base_m(c2_bits[n][m],pxp.base)

c1_basis = np.zeros((pxp.dim,len(c1_refs)))
c2_basis = np.zeros((pxp.dim,len(c2_refs)))
for n in range(0,len(c1_refs)):
    temp = np.zeros(pxp.dim)
    temp2 = np.zeros(pxp.dim)
    for m in range(0,np.size(c1_refs[n],axis=0)):
        temp[pxp.keys[c1_refs[n][m]]] = 1
        temp2[pxp.keys[c2_refs[n][m]]] = 1
    temp = temp / np.power(np.vdot(temp,temp),0.5)
    temp2 = temp2 / np.power(np.vdot(temp2,temp2),0.5)
    c1_basis[:,n] = temp
    c2_basis[:,n] = temp2

z1=zm_state(3,1,pxp)
z2=zm_state(3,1,pxp,1)
z3=zm_state(3,1,pxp,2)
z0=ref_state(0,pxp)
print(z1.ref,z2.ref,z3.ref,z0.ref)

coupling_basis = np.zeros((pxp.dim,4))
coupling_basis[:,0] = z1.prod_basis()
coupling_basis[:,1] = z2.prod_basis()
coupling_basis[:,2] = z3.prod_basis()
coupling_basis[:,3] = z0.prod_basis()

basis = np.hstack((c1_basis,coupling_basis))
basis = np.hstack((basis,np.flip(c2_basis,axis=1)))

pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])
H=spin_Hamiltonian(pxp,"x",pxp_syms)
z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    overlap = eig_overlap(z,H,k[n]).eval()
    plt.scatter(H.sector.eigvalues(k[n]),overlap,color="blue")
H.gen()
H.sector.find_eig()

# eig_overlap(z,H).plot()
# plt.show()

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
# plt.matshow(np.abs(H_rot))
# plt.show()

print(H.sector.eigvalues())
e,u = np.linalg.eigh(H_rot)
print(e)

psi = np.zeros(np.size(e))
psi[0] = 1
psi_energy = np.conj(np.dot(u,psi))
z_energy=u[0,:]
overlap = np.log10(np.abs(psi_energy)**2)
plt.scatter(e,overlap,marker="x",s=100,color="red",label="subcube")

# fsa
z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits,pxp)
fsa_basis = np.zeros((pxp.dim,len(hamming_sectors)))
for n in range(0,np.size(fsa_basis,axis=1)):
    temp = np.zeros(pxp.dim)
    for m in range(0,np.size(hamming_sectors[n],axis=0)):
        temp = temp + ref_state(hamming_sectors[n][m],pxp).prod_basis()
    temp = temp / np.power(np.vdot(temp,temp),0.5)
    fsa_basis[:,n] = temp
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e_fsa,u_fsa = np.linalg.eigh(H_fsa)

psi_fsa = np.zeros(np.size(e_fsa))
psi_fsa[0] = 1
psi_energy = np.conj(np.dot(u_fsa,psi_fsa))
z_energy=u[0,:]
overlap = np.log10(np.abs(psi_energy)**2)
plt.scatter(e_fsa,overlap,marker="x",s=100,label="FSA")
plt.legend()
plt.show()

#overlap of approximate scars with act scars
u_comp_basis = np.dot(basis,u)
m_overlaps = np.zeros(np.size(e,axis=0))
for n in range(0,np.size(e,axis=0)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        overlap = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if overlap > max_overlap:
            max_overlap = overlap
    m_overlaps[n] = max_overlap
plt.scatter(e,m_overlaps,label="SubCube Basis")

#overlap of approximate scars with act scars
u_comp_basis = np.dot(fsa_basis,u_fsa)
m_overlaps_fsa = np.zeros(np.size(e_fsa,axis=0))
for n in range(0,np.size(e_fsa,axis=0)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        overlap = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if overlap > max_overlap:
            max_overlap = overlap
    m_overlaps_fsa[n] = max_overlap
print(m_overlaps)
plt.scatter(e_fsa,m_overlaps_fsa,marker="x",label="Fsa Basis")
plt.legend()
plt.show()
