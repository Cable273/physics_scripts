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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state, gram_schmidt,gen_krylov_basis

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 4
base=3
pxp = unlocking_System([0],"periodic",base,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[parity(pxp),translational(pxp)])

z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
block_refs1 = pxp_syms.find_block_refs(k[0])
block_refs2 = pxp_syms.find_block_refs(k[1])
print(np.size(block_refs1)+np.size(block_refs2))

H = clock_Hamiltonian(pxp,pxp_syms)
H.gen()
H.sector.find_eig()

#krylov basis
krylov_dim = 2*pxp.N
krylov_basis = gen_krylov_basis(H.sector.matrix(),krylov_dim,z.prod_basis(),pxp,orth="qr")

#FSA basis
# P+P on even sites
Ipe = Hamiltonian(pxp,pxp_syms)
Ipe.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ipe.model = np.array([[0,1,0]])
Ipe.model_coef = np.array([1])
Ipe.gen(parity=1)
Imo = Hamiltonian(pxp,pxp_syms)
Imo.site_ops[1] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Imo.model = np.array([[0,1,0]])
Imo.model_coef = np.array([1])
Imo.gen(parity=0)
Ip = H_operations.add(Ipe,Imo,np.array([1,-1]))
Ip = Ip.sector.matrix()
Im = np.conj(np.transpose(Ip))

Kpe = Hamiltonian(pxp,pxp_syms)
Kpe.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kpe.model = np.array([[0,1,0]])
Kpe.model_coef = np.array([1])
Kpe.gen(parity=1)
Kmo = Hamiltonian(pxp,pxp_syms)
Kmo.site_ops[1] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kmo.model = np.array([[0,1,0]])
Kmo.model_coef = np.array([1])
Kmo.gen(parity=0)
Kp = H_operations.add(Kpe,Kmo,np.array([1,-1]))
Kp = Kp.sector.matrix()
Km = np.conj(np.transpose(Kp))

Lpe = Hamiltonian(pxp,pxp_syms)
Lpe.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lpe.model = np.array([[0,1,0]])
Lpe.model_coef = np.array([1])
Lpe.gen(parity=1)
Lmo = Hamiltonian(pxp,pxp_syms)
Lmo.site_ops[1] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lmo.model = np.array([[0,1,0]])
Lmo.model_coef = np.array([1])
Lmo.gen(parity=0)
Lp = H_operations.add(Ipe,Imo,np.array([1,-1]))
Lp = Lp.sector.matrix()
Lm = np.conj(np.transpose(Lp))

raising_lowering_ops = dict()
raising_lowering_ops[0] = Ip
raising_lowering_ops[1] = Kp
raising_lowering_ops[2] = Lp
raising_lowering_ops[3] = Im
raising_lowering_ops[4] = Km
raising_lowering_ops[5] = Lm

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
g3 = 1/2 * com(Ip,Im)
g8 = 1/np.power(3,0.5)*com(Kp,Km)-g3

z=zm_state(2,1,pxp)
fsa_basis = np.zeros(pxp.dim)
fsa_basis = np.vstack((fsa_basis,z.prod_basis()))
current_states = fsa_basis
from Diagnostics import print_wf
pbar=ProgressBar()
for n in pbar(range(0,pxp.N)):
    print("\n")
    new_states = [np.zeros((pxp.dim))]
    for state in range(0,np.size(current_states,axis=0)):
        for op in range(0,len(raising_lowering_ops)):
            temp = np.dot(raising_lowering_ops[op],current_states[state])
            if np.abs(np.vdot(temp,temp))>1e-5:
                temp = temp / np.power(np.vdot(temp,temp),0.5)
                already_found = 0
                for i in range(0,np.size(fsa_basis,axis=0)):
                    overlap = np.abs(np.vdot(temp,fsa_basis[i,:]))
                    if overlap == 1:
                        already_found = 1
                        break
                if already_found == 0:
                    new_states = np.vstack((new_states,temp))
    if np.size(new_states,axis=0)!=1:
        new_states = np.unique(np.delete(new_states,0,axis=0),axis=0)
        fsa_basis = np.vstack((fsa_basis,new_states))
        current_states = new_states
    else:
        break

fsa_basis = np.transpose(np.delete(fsa_basis,0,axis=0))
gs = gram_schmidt(fsa_basis)
gs.ortho()
fsa_basis = gs.ortho_basis
print(np.shape(fsa_basis))

# z1=zm_state(2,1,pxp)
# z2=zm_state(2,2,pxp)
# fsa_basis1 = z1.prod_basis()
# fsa_basis2 = z2.prod_basis()
# current_state1 = fsa_basis1
# current_state2 = fsa_basis2
# for n in range(0,pxp.N):
    # next_state1 = np.dot(Ip,current_state1)
    # next_state2 = np.dot(Kp,current_state2)
    # next_state1 = next_state1/np.power(np.vdot(next_state1,next_state1),0.5)
    # next_state2 = next_state2/np.power(np.vdot(next_state2,next_state2),0.5)
    # fsa_basis1 = np.vstack((fsa_basis1,next_state1))
    # fsa_basis2 = np.vstack((fsa_basis2,next_state2))
    # current_state1 = next_state1
    # current_state2 = next_state2
# fsa_basis1 = np.transpose(fsa_basis1)
# fsa_basis2 = np.transpose(fsa_basis2)
# print(np.shape(fsa_basis1))
# print(np.shape(fsa_basis2))

# fsa_basis = np.hstack((fsa_basis1,fsa_basis2))
# fsa_basis,temp = np.linalg.qr(fsa_basis)
# gs = gram_schmidt(fsa_basis)
# gs.ortho()
# fsa_basis = gs.ortho_basis
print(np.shape(fsa_basis))


# # # # # #project hamiltonians
H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
for n in range(0,np.size(H_fsa,axis=0)):
    for m in range(0,np.size(H_fsa,axis=0)):
        if np.abs(H_fsa[n,m])<1e-10:
            H_fsa[n,m] = 0
    
from Calculations import plot_adjacency_graph
plot_adjacency_graph(np.abs(H_fsa))
plt.show()

plt.matshow(np.abs(H_fsa))
plt.show()

e_krylov,u_krylov = np.linalg.eigh(H_krylov)
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
print(e_krylov)
print(e_fsa)

overlap_exact = eig_overlap(z,H).eval()
overlap_krylov = np.log10(np.abs(u_krylov[0,:])**2)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)

eig_overlap(z,H).plot()
# plt.scatter(H.sector.eigvalues(),overlap_exact)
plt.scatter(e_krylov,overlap_krylov,marker="x",s=200,color="cyan",label="Krylov",alpha=0.6)
plt.scatter(e_fsa,overlap_fsa,marker="s",s=100,label="FSA",color="red",alpha=0.6)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()

u_krylov_comp = np.dot(krylov_basis,u_krylov)
u_fsa_comp = np.dot(fsa_basis,u_fsa)

exact_overlap_fsa = np.zeros(np.size(e_fsa))
for n in range(0,np.size(u_fsa_comp,axis=1)):
    max_overlap_fsa = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp_fsa = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))
        if temp_fsa> max_overlap_fsa:
            max_overlap_fsa = temp_fsa
    exact_overlap_fsa[n] = max_overlap_fsa

exact_overlap_krylov = np.zeros(np.size(e_krylov))
for n in range(0,np.size(u_krylov_comp,axis=1)):
    max_overlap_krylov = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp_krylov = np.abs(np.vdot(u_krylov_comp[:,n],H.sector.eigvectors()[:,m]))
        if temp_krylov> max_overlap_krylov:
            max_overlap_krylov = temp_krylov
    exact_overlap_krylov[n] = max_overlap_krylov

plt.plot(e_krylov,exact_overlap_krylov,marker="s",label="Krylov")
plt.plot(e_fsa,exact_overlap_fsa,marker="s",label="FSA")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()
        
    
