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

from Hamiltonian_Classes import *
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import *
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

N = 18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=3)])

V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,0],[1,0]])
V1_ops[0].model = np.array([[0,0,1,0]])
V1_ops[0].model_coef = np.array([1])
V1_ops[0].gen(uc_size=3,uc_pos=2)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,0],[1,0]])
V1_ops[1].model = np.array([[0,1,0,0]])
V1_ops[1].model_coef = np.array([1])
V1_ops[1].gen(uc_size=3,uc_pos=1)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[0,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
V1_ops[2].gen(uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[0,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
V1_ops[3].gen(uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,0],[1,0]])
V2_ops[0].model = np.array([[0,1,0,0]])
V2_ops[0].model_coef = np.array([1])
V2_ops[0].gen(uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,0],[1,0]])
V2_ops[1].model = np.array([[0,0,1,0]])
V2_ops[1].model_coef = np.array([1])
V2_ops[1].gen(uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,0],[1,0]])
V3_ops[0].site_ops[2] = np.array([[0,1],[0,0]])
V3_ops[0].model = np.array([[0,2,1,2,0]])
V3_ops[0].model_coef = np.array([1])
V3_ops[0].gen(uc_size=3,uc_pos=2)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,0],[1,0]])
V3_ops[1].site_ops[2] = np.array([[0,1],[0,0]])
V3_ops[1].model = np.array([[0,2,1,2,0]])
V3_ops[1].model_coef = np.array([1])
V3_ops[1].gen(uc_size=3,uc_pos=0)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

Hp0_ops = dict()
Hp0_ops[0] = Hamiltonian(pxp,pxp_syms)
Hp0_ops[0].site_ops[1] = np.array([[0,1],[0,0]])
Hp0_ops[0].model = np.array([[0,1,0]])
Hp0_ops[0].model_coef = np.array([1])
Hp0_ops[0].gen(uc_size=3,uc_pos=2)

Hp0_ops[1] = Hamiltonian(pxp,pxp_syms)
Hp0_ops[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp0_ops[1].model = np.array([[0,1,0]])
Hp0_ops[1].model_coef = np.array([1])
Hp0_ops[1].gen(uc_size=3,uc_pos=0)

Hp0_ops[2] = Hamiltonian(pxp,pxp_syms)
Hp0_ops[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp0_ops[2].model = np.array([[0,1,0]])
Hp0_ops[2].model_coef = np.array([1])
Hp0_ops[2].gen(uc_size=3,uc_pos=1)

Hp0 = Hp0_ops[0]
for n in range(1,len(Hp0_ops)):
    Hp0=H_operations.add(Hp0,Hp0_ops[n],np.array([1,1]))

coef = np.array([0.18243653,-0.10390499,0.054452])
Hp = H_operations.add(Hp0,V1,np.array([1,coef[0]]))
Hp = H_operations.add(Hp,V2,np.array([1,coef[1]]))
Hp = H_operations.add(Hp,V3,np.array([1,coef[2]]))
# Hp = Hp0.sector.matrix()
Hp=Hp.sector.matrix()
Hm = np.conj(np.transpose(Hp))

#generate perturbed Hamiltonian
V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,0,0]])
V1_ops[0].model_coef = np.array([1])
V1_ops[0].gen(uc_size=3,uc_pos=1)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[1].model = np.array([[0,0,1,0]])
V1_ops[1].model_coef = np.array([1])
V1_ops[1].gen(uc_size=3,uc_pos=2)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
V1_ops[2].gen(uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
V1_ops[3].gen(uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,0,1,0]])
V2_ops[0].model_coef = np.array([1])
V2_ops[0].gen(uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,0,0]])
V2_ops[1].model_coef = np.array([1])
V2_ops[1].gen(uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,1,1,1,0]])
V3_ops[0].model_coef = np.array([1])
V3_ops[0].gen(uc_size=3,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,1,1,0]])
V3_ops[1].model_coef = np.array([1])
V3_ops[1].gen(uc_size=3,uc_pos=2)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()
H = H_operations.add(H0,V1,np.array([1,coef[0]]))
H = H_operations.add(H,V2,np.array([1,coef[1]]))
H = H_operations.add(H,V3,np.array([1,coef[2]]))
# H=H0
H.sector.find_eig()

temp = Hp + Hm
print((np.abs(temp-H.sector.matrix())<1e-5).all())

z=zm_state(3,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
fsa_dim = int(2*pxp.N/3)
for n in range(0,fsa_dim):
    next_state = np.dot(Hp,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)

from Calculations import gen_krylov_basis
krylov_basis = gen_krylov_basis(H.sector.matrix(),int(2*pxp.N/3),z.prod_basis(),pxp,orth="gs")

H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
e,u = np.linalg.eigh(H_fsa)
ek,uk = np.linalg.eigh(H_krylov)
fsa_overlap = np.log10(np.abs(u[0,:])**2)
krylov_overlap = np.log10(np.abs(uk[0,:])**2)

eig_overlap(z,H).plot()
plt.scatter(ek,krylov_overlap,marker="x",s=100,color="red",label="Krylov")
plt.scatter(e,fsa_overlap,marker="D",s=100,alpha=0.6,label="FSA")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \psi \vert E \rangle \vert^2)$")
plt.title(r"$PXP + \lambda_i V_i$, $Z_3$ Scar approximations, $N=$"+str(pxp.N))
plt.show()

u_comp = np.dot(fsa_basis,u)
u_compk = np.dot(fsa_basis,uk)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(e,axis=0)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp = np.abs(np.vdot(u_comp[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap

exact_overlapk = np.zeros(np.size(ek))
for n in range(0,np.size(ek,axis=0)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp = np.abs(np.vdot(u_compk[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlapk[n] = max_overlap

plt.plot(ek,exact_overlapk,marker="s",label="Krylov")
plt.plot(e,exact_overlap,marker="s",label="FSA")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"$PXP + \lambda_i V_i$, $Z_3$ Scar approximations, $N=$"+str(pxp.N))
plt.show()
        

