#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
file_dir = '/localhome/pykb/Python/Tensor_Train/'
sys.path.append(file_dir)
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *

file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System
from Symmetry_Classes import *
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state
import math

N=18
D=2
d=2

#convert MPS -> wf array
pxp = unlocking_System([0],"periodic",d,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=3)])
z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
U = dict()
for n in range(0,np.size(k,axis=0)):
    U[int(k[n])] = pxp_syms.basis_transformation(k[n])

wf = np.load("./z3,entangled_MPS_coef,"+str(pxp.N)+".npy")

#project wf to symmetry basis
wf_sym = dict()
for n in range(0,np.size(k,axis=0)):
    wf_sym[int(k[n])] = np.dot(np.conj(np.transpose(U[int(k[n])])),wf)

# dynamics + fidelity
V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,0,0]])
V1_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=1)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[1].model = np.array([[0,0,1,0]])
V1_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[2].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V1_ops[3].gen(k_vec=k[n],uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,0,1,0]])
V2_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V2_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,0,0]])
V2_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V2_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,1,1,1,0]])
V3_ops[0].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V3_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,1,1,0]])
V3_ops[1].model_coef = np.array([1])
for n in range(0,np.size(k)):
    V3_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
for n in range(0,np.size(k,axis=0)):
    H0.gen(k[n])
coef = np.array([0.18243653,-0.10390499,0.054452])
H = H_operations.add(H0,V1,np.array([1,coef[0]]))
H = H_operations.add(H,V2,np.array([1,coef[1]]))
H = H_operations.add(H,V3,np.array([1,coef[2]]))
# H=H0
for n in range(0,np.size(k,axis=0)):
    H.sector.find_eig(k[n])

e = H.sector.eigvalues(k[0])
u = H.sector.eigvectors(k[0])

psi_energy = np.dot(np.conj(np.transpose(u)),wf_sym[0])
eigenvalues = e
overlap = np.log10(np.abs(psi_energy)**2)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
    
plt.scatter(eigenvalues,overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"PXP, Eigenstate overlap with K=3 Entangled MPS, N=18")
plt.show()

t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
plt.title(r"PXP, K=3 Entangled MPS Fidelity, N=18")
plt.title(r"$PXP+\lambda_i V_i, K=3$ Entangled MPS Fidelity, N="+str(pxp.N))
plt.plot(t,f,label="K=3 Entangled MPS")

z3=zm_state(3,1,pxp)
f = fidelity(z3,H,"use syms").eval(t,z3)
plt.plot(t,f,label="Z3")
plt.legend()
plt.show()


