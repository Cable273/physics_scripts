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

N = 8
base=3
pxp = unlocking_System([0],"periodic",base,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H = clock_Hamiltonian(pxp,pxp_syms)
# H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

z=zm_state(2,1,pxp)

#krylov basis
krylov_dim = 2*pxp.N
krylov_basis = gen_krylov_basis(H.sector.matrix(),krylov_dim,z.prod_basis(),pxp,orth="qr")

#FSA basis
# P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
# pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.site_ops[1] = np.array([[0,1j,0],[0,0,1j],[1j,0,0]])
pe.model = np.array([[0,1,0]])
pe.model_coef = np.array([1])
pe.gen(parity=1)
#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
# mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.site_ops[1] = np.array([[0,0,-1j],[-1j,0,0],[0,-1j,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=0)
#Raising op
Hp = H_operations.add(pe,mo,np.array([1,1]))
Hp = Hp.sector.matrix()
Hm = np.conj(np.transpose(Hp))
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,krylov_dim):
    next_state = np.dot(Hm,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
# fsa_basis,temp = np.linalg.qr(fsa_basis)
gs = gram_schmidt(fsa_basis)
gs.ortho()
fsa_basis = gs.ortho_basis

#FSA basis 0->1 clock
# P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
# pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.site_ops[1] = np.array([[0,0,0],[-1j,0,1j],[1j,0,0]])
pe.model = np.array([[0,1,0]])
pe.model_coef = np.array([1])
pe.gen(parity=1)
#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
# mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.site_ops[1] = np.array([[0,1j,-1j],[0,0,0],[0,-1j,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=0)
#Raising op
Hp = H_operations.add(pe,mo,np.array([1,1]))
Hp = Hp.sector.matrix()
Hm = np.conj(np.transpose(Hp))
fsa_basis2 = z.prod_basis()
current_state = fsa_basis2
for n in range(0,krylov_dim):
    next_state = np.dot(Hm,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis2 = np.vstack((fsa_basis2,next_state))
    current_state = next_state
fsa_basis2 = np.transpose(fsa_basis2)
fsa_basis2,temp = np.linalg.qr(fsa_basis2)
# gs = gram_schmidt(fsa_basis)
# gs.ortho()
# fsa_basis = gs.ortho_basis

#gFSA basis, raising operator from single site krylov
C = H.site_ops[1]
init_state0 = np.array([1,0,0])
init_state1 = np.array([0,1,0])
# init_state0 = np.array([1,0])
# init_state1 = np.array([0,1])

small_krylov_basis0 = gen_krylov_basis(C,3,init_state0,pxp,orth="gs")
print(small_krylov_basis0)
# small_krylov_basis1 = gen_krylov_basis(C,3,init_state1,pxp,orth="gs")
C_krylov0 = np.dot(np.conj(np.transpose(small_krylov_basis0)),np.dot(C,small_krylov_basis0))
print(C_krylov0)
# C_krylov1 = np.dot(np.conj(np.transpose(small_krylov_basis1)),np.dot(C,small_krylov_basis1))
sp_krylov = np.diag(np.diag(C_krylov0,1),1)
sm_krylov = np.diag(np.diag(C_krylov0,-1),-1)
sp = np.dot(small_krylov_basis0,np.dot(sp_krylov,np.conj(np.transpose(small_krylov_basis0))))
sm = np.dot(small_krylov_basis0,np.dot(sm_krylov,np.conj(np.transpose(small_krylov_basis0))))
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
temp = com(sp,sm)
print(temp)
# print(sp)
# print(sm)
# print(sp+sm)
# print(C)
# P+P on even sites
gpe = Hamiltonian(pxp,pxp_syms)
gpe.site_ops[1] = sp
gpe.model = np.array([[0,1,0]])
gpe.model_coef = np.array([1])
gpe.gen(parity=1)
#P-P on odd sites
gmo = Hamiltonian(pxp,pxp_syms)
gmo.site_ops[1] = sm
gmo.model = np.array([[0,1,0]])
gmo.model_coef = np.array([1])
gmo.gen(parity=0)
#Raising op
gHp = H_operations.add(gpe,gmo,np.array([1,1]))
gfsa_basis = z.prod_basis()
current_state = gfsa_basis
for n in range(0,krylov_dim):
    next_state = np.dot(gHp.sector.matrix(),current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    gfsa_basis = np.vstack((gfsa_basis,next_state))
    current_state = next_state
gfsa_basis = np.transpose(gfsa_basis)
gfsa_basis,temp = np.linalg.qr(gfsa_basis)
# gs = gram_schmidt(gfsa_basis)
# gs.ortho()
# gfsa_basis = gs.ortho_basis

#project hamiltonians
H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
H_fsa2 = np.dot(np.conj(np.transpose(fsa_basis2)),np.dot(H.sector.matrix(),fsa_basis2))
H_gfsa = np.dot(np.conj(np.transpose(gfsa_basis)),np.dot(H.sector.matrix(),gfsa_basis))
plt.matshow(np.abs(H_gfsa))
plt.show()

e_krylov,u_krylov = np.linalg.eigh(H_krylov)
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
e_fsa2,u_fsa2 = np.linalg.eigh(H_fsa2)
e_gfsa,u_gfsa = np.linalg.eigh(H_gfsa)
print(e_krylov)
print(e_fsa)
print(e_gfsa)

overlap_krylov = np.log10(np.abs(u_krylov[0,:])**2)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)
overlap_fsa2 = np.log10(np.abs(u_fsa2[0,:])**2)
overlap_gfsa = np.log10(np.abs(u_gfsa[0,:])**2)

e_gfsa2 = np.load("./fsa,sub_raising,e,3,12.npy")
overlap_gfsa2 = np.load("./fsa,sub_raising,overlap,3,12.npy")
exact_overlap_gfsa2 = np.load("./fsa,sub_raising,exact_overlap,3,12.npy")

eig_overlap(z,H).plot()
plt.scatter(e_krylov,overlap_krylov,marker="x",s=200,color="cyan",label="Krylov",alpha=0.6)
plt.scatter(e_fsa,overlap_fsa,marker="s",s=100,label="FSA (CW/ACW)",color="red",alpha=0.6)
plt.scatter(e_fsa2,overlap_fsa2,marker="s",s=100,label=r"FSA ($0\rightarrow 1$)",color="cyan",alpha=0.6)
plt.scatter(e_gfsa,overlap_gfsa,marker="D",s=100,label="FSA (Site Krylov)",color="green",alpha=0.6)
plt.scatter(e_gfsa2,overlap_gfsa2,marker="D",s=100,label="FSA (Sublattice Krylov)",alpha=0.6)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()

u_krylov_comp = np.dot(krylov_basis,u_krylov)
u_fsa_comp = np.dot(fsa_basis,u_fsa)
u_fsa_comp2 = np.dot(fsa_basis2,u_fsa2)
u_gfsa_comp = np.dot(gfsa_basis,u_gfsa)

exact_overlap_krylov = np.zeros(np.size(e_krylov))
exact_overlap_fsa = np.zeros(np.size(e_fsa))
exact_overlap_fsa2 = np.zeros(np.size(e_fsa2))
exact_overlap_gfsa = np.zeros(np.size(e_gfsa))
for n in range(0,np.size(u_gfsa_comp,axis=1)):
    max_overlap_krylov = 0
    max_overlap_fsa = 0
    max_overlap_fsa2 = 0
    max_overlap_gfsa = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp_krylov = np.abs(np.vdot(u_krylov_comp[:,n],H.sector.eigvectors()[:,m]))
        temp_fsa = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))
        temp_fsa2 = np.abs(np.vdot(u_fsa_comp2[:,n],H.sector.eigvectors()[:,m]))
        temp_gfsa = np.abs(np.vdot(u_gfsa_comp[:,n],H.sector.eigvectors()[:,m]))
        if temp_krylov> max_overlap_krylov:
            max_overlap_krylov = temp_krylov
        if temp_fsa> max_overlap_fsa:
            max_overlap_fsa = temp_fsa
        if temp_fsa2> max_overlap_fsa2:
            max_overlap_fsa2 = temp_fsa2
        if temp_gfsa> max_overlap_gfsa:
            max_overlap_gfsa = temp_gfsa
    exact_overlap_krylov[n] = max_overlap_krylov
    exact_overlap_fsa[n] = max_overlap_fsa
    exact_overlap_fsa2[n] = max_overlap_fsa2
    exact_overlap_gfsa[n] = max_overlap_gfsa

plt.plot(e_krylov,exact_overlap_krylov,marker="s",label="Krylov")
plt.plot(e_fsa,exact_overlap_fsa,marker="s",label="FSA (CW/ACW)")
plt.plot(e_fsa2,exact_overlap_fsa2,marker="s",label=r"FSA ($0\rightarrow 1$)")
plt.plot(e_gfsa,exact_overlap_gfsa,marker="D",alpha=0.6,label="FSA (Site Krylov)")
plt.plot(e_gfsa2,exact_overlap_gfsa2,marker="D",alpha=0.6,label="FSA (Sublattice Krylov)")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()
        
    
