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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state, gram_schmidt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 12
base=3
pxp = unlocking_System([0],"periodic",base,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H = clock_Hamiltonian(pxp,pxp_syms)
# H = spin_Hamiltonian(pxp,"x",pxp_syms)
# H = clock_Hamiltonian(pxp,pxp_syms)
H.gen()
H.sector.find_eig()

z=zm_state(2,1,pxp)
krylov_dim = 2*pxp.N

def gen_krylov_basis(H,dim,init_state,system,orth=False):
    krylov_basis = init_state.prod_basis()
    current_state = krylov_basis
    for n in range(0,dim):
        next_state = np.dot(H,current_state)
        next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
        krylov_basis = np.vstack((krylov_basis,next_state))
        current_state = next_state
    krylov_basis = np.transpose(krylov_basis)
    if orth is False:
        return krylov_basis
    else:
        if orth == "qr":
            krylov_basis,temp = np.linalg.qr(krylov_basis)
        else:
            gs = gram_schmidt(krylov_basis)
            gs.ortho()
            krylov_basis = gs.ortho_basis
    return krylov_basis

#krylov basis
krylov_basis = gen_krylov_basis(H.sector.matrix(),krylov_dim,z,pxp,orth="qr")

#FSA basis
# P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
# pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.site_ops[1] = np.array([[0,1j,0],[0,0,1j],[1j,0,0]])
# pe.site_ops[1] = np.array([[0,1j,-1j],[0,0,1j],[0,0,0]])
pe.model = np.array([[0,1,0]])
pe.model_coef = np.array([1])
pe.gen(parity=1)

#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
# mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.site_ops[1] = np.array([[0,0,-1j],[-1j,0,0],[0,-1j,0]])
# mo.site_ops[1] = np.array([[0,0,0],[-1j,0,0],[1j,-1j,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=0)

Hp = H_operations.add(pe,mo,np.array([1,1]))

fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,krylov_dim):
    next_state = np.dot(Hp.sector.matrix(),current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
fsa_basis,temp = np.linalg.qr(fsa_basis)
# gs = gram_schmidt(fsa_basis)
# gs.ortho()
# fsa_basis = gs.ortho_basis

#general fsa (krylov on sublattices)
pxp_half = unlocking_System([0,1,2],"periodic",base,int(N/2))
pxp_half.gen_basis()
pxp_half_syms=model_sym_data(pxp_half,[translational(pxp_half)])
krylov_half_dim = 2*pxp_half.N

#need linear opp which combines |n>_e x |m>_o
def join_basis_states(n,m,system,double_system):
    even_state_bits = system.basis[n]
    odd_state_bits = system.basis[m]
    full_bits = np.zeros(2*np.size(even_state_bits))
    c_even = 0
    c_odd = 0
    for n in range(0,np.size(full_bits,axis=0)):
        if n % 2 == 0:
            full_bits[n] = even_state_bits[c_even]
            c_even = c_even + 1
        else:
            full_bits[n] = odd_state_bits[c_odd]
            c_odd = c_odd + 1
    full_bits_ref = bin_to_int_base_m(full_bits,double_system.base)
    if full_bits_ref in double_system.basis_refs:
        return full_bits_ref
    else:
        return None

#to join two lin comb of the basis states:
M_basis_map = np.zeros((pxp_half.dim,pxp_half.dim))
for n in range(0,pxp_half.dim):
    for m in range(0,pxp_half.dim):
        M_basis_map[n,m] = join_basis_states(n,m,pxp_half,pxp)
M_basis_map = np.ndarray.flatten(M_basis_map)

def half_rep_to_full(state_half_rep,M_basis_map,system,double_system):
    M_coef = np.ndarray.flatten(state_half_rep)
    new_state = np.zeros(double_system.dim,dtype=complex)
    for n in range(0,np.size(M_coef,axis=0)):
        if math.isnan(M_basis_map[n]) is False:
            key = double_system.keys[M_basis_map[n]]
            if key is not None:
                new_state[key] = new_state[key] + M_coef[n]
    return new_state

def full_rep_to_half(state,M_basis_map,system,double_system):
    state_half_rep = np.zeros((np.size(M_basis_map)),dtype=complex)
    for n in range(0,np.size(M_basis_map,axis=0)):
        if math.isnan(M_basis_map[n]) is False:
            state_half_rep[n] = state[double_system.keys[M_basis_map[n]]]
    state_half_rep = np.reshape(state_half_rep,(system.dim,system.dim))
    return state_half_rep
        

def join_states(psi1,psi2,M_basis_map,system,double_system):
    M_coef = np.zeros((system.dim,system.dim),dtype=complex)
    for n in range(0,system.dim):
        for m in range(0,system.dim):
            M_coef[n,m] = psi1[n]*psi2[m]
    new_state = half_rep_to_full(M_coef,M_basis_map,system,double_system)
    return new_state

# H_half = spin_Hamiltonian(pxp_half,"x",pxp_half_syms)
H_half = clock_Hamiltonian(pxp_half,pxp_half_syms)
H_half.gen()
# z_even = ref_state(np.max(pxp_half.basis_refs),pxp_half).prod_basis()
# z_odd = ref_state(0,pxp_half).prod_basis()

z_even = ref_state(np.max(pxp_half.basis_refs),pxp_half)
z_odd = ref_state(0,pxp_half)
z_even_prod = z_even.prod_basis()
z_odd_prod = z_odd.prod_basis()
print(z_even.bits)
print(z_odd.bits)

state_half_rep = np.zeros((pxp_half.dim,pxp_half.dim))
for n in range(0,np.size(state_half_rep,axis=0)):
    for m in range(0,np.size(state_half_rep,axis=0)):
        state_half_rep[n,m] = z_even_prod[n]*z_odd_prod[m]

#find Hp sublattice operators in sublattice H space from upper tridiagonal of krylov in sublattice H
even_krylov_basis = gen_krylov_basis(H_half.sector.matrix(),krylov_half_dim,z_even,pxp_half,orth="gs")
odd_krylov_basis = gen_krylov_basis(H_half.sector.matrix(),krylov_half_dim,z_odd,pxp_half,orth="gs")
proj_H_even = np.dot(np.conj(np.transpose(even_krylov_basis)),np.dot(H_half.sector.matrix(),even_krylov_basis))
plt.matshow(np.abs(proj_H_even))
proj_Hp_even = np.diag(np.diag(proj_H_even,1),1)
plt.matshow(np.abs(proj_Hp_even))
plt.show()

Hp_zeros = np.dot(even_krylov_basis,np.dot(proj_Hp_even,np.conj(np.transpose(even_krylov_basis))))
Hp_ones = np.dot(odd_krylov_basis,np.dot(proj_Hp_even,np.conj(np.transpose(odd_krylov_basis))))

plt.matshow(np.abs(Hp_zeros))
plt.show()

# Hp_zeros = np.zeros((pxp_half.dim,pxp_half.dim),dtype=complex)
# for n in range(0,pxp_half.dim):
    # for m in range(0,n):
        # Hp_zeros[n,m] = H_half.sector.matrix()[n,m]
# Hp_ones = np.conj(np.transpose(Hp_zeros))
# plt.matshow(np.abs(Hp_ones))
# plt.matshow(np.abs(Hp_zeros))
# plt.show()

# Hp_temp = Hamiltonian(pxp_half)
# Hp_temp.site_ops[1] = np.array([[0,1],[0,0]])
# # Hp_temp.site_ops[1] = np.array([[0,1j,0],[0,0,1j],[1j,0,0]])
# # Hp_temp.site_ops[1] = np.array([[0,1j,-1j],[-1j,0,0],[1j,0,0]])
# # Hp_temp.site_ops[1] = np.array([[0,1j,-1j],[0,0,1j],[0,0,0]])
# Hp_temp.model = np.array([[1]])
# Hp_temp.model_coef = np.array([1])
# Hp_temp.gen()

# Hm_temp = Hamiltonian(pxp_half)
# Hm_temp.site_ops[1] = np.array([[0,0],[1,0]])
# # Hm_temp.site_ops[1] = np.array([[0,0,-1j],[-1j,0,0],[0,-1j,0]])
# # Hm_temp.site_ops[1] = np.array([[0,0,-1j],[0,0,1j],[1j,-1j,0]])
# # Hm_temp.site_ops[1] = np.array([[0,0,0],[-1j,0,0],[1j,-1j,0]])
# Hm_temp.model = np.array([[1]])
# Hm_temp.model_coef = np.array([1])
# Hm_temp.gen()
# print((np.abs(Hp_temp.sector.matrix()+Hm_temp.sector.matrix()-H_half.sector.matrix())<1e-5).all())
# plt.matshow(np.abs(Hp_temp.sector.matrix()))
# plt.show()

def gsa_Hp(state_half_rep):
    C=np.einsum('ij,ui->uj',state_half_rep,Hp_ones)+np.einsum('ij,uj->iu',state_half_rep,Hp_zeros)
    # C=np.einsum('ij,ui->uj',state_half_rep,H_half.sector.matrix())+np.einsum('ij,uj->iu',state_half_rep,H_half.sector.matrix())
    # C=np.einsum('ij,ui->uj',state_half_rep,Hp_temp.sector.matrix())+np.einsum('ij,uj->iu',state_half_rep,Hm_temp.sector.matrix())
    # C=np.einsum('ij,ui->uj',state_half_rep,Hm_temp.sector.matrix())+np.einsum('ij,uj->iu',state_half_rep,Hp_temp.sector.matrix())
    temp = half_rep_to_full(C,M_basis_map,pxp_half,pxp)
    C = full_rep_to_half(temp,M_basis_map,pxp_half,pxp)
    return C
    
gfsa_basis = join_states(z_even_prod,z_odd_prod,M_basis_map,pxp_half,pxp)
current_state_half_rep = full_rep_to_half(gfsa_basis,M_basis_map,pxp_half,pxp)

#need mapping from full basis to even/odd sub basis
for n in range(0,krylov_dim):
    next_state_half_rep = gsa_Hp(current_state_half_rep)
    next_state_full_rep = half_rep_to_full(next_state_half_rep,M_basis_map,pxp_half,pxp)
    gfsa_basis = np.vstack((gfsa_basis,next_state_full_rep))

    # gs = gram_schmidt(np.transpose(gfsa_basis))
    # gs.ortho()
    # gfsa_basis = np.transpose(gs.ortho_basis)

    current_state_half_rep = next_state_half_rep
gfsa_basis = np.transpose(gfsa_basis)
# gfsa_basis,temp = np.linalg.qr(gfsa_basis)
gs = gram_schmidt(gfsa_basis)
gs.ortho()
gfsa_basis = gs.ortho_basis

#project hamiltonians
H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
H_gfsa = np.dot(np.conj(np.transpose(gfsa_basis)),np.dot(H.sector.matrix(),gfsa_basis))

e_krylov,u_krylov = np.linalg.eigh(H_krylov)
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
e_gfsa,u_gfsa = np.linalg.eigh(H_gfsa)
print(e_krylov)
print(e_fsa)
print(e_gfsa)

overlap_krylov = np.log10(np.abs(u_krylov[0,:])**2)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)
overlap_gfsa = np.log10(np.abs(u_gfsa[0,:])**2)

eig_overlap(z,H).plot()
# plt.scatter(e_fsa,overlap_fsa,marker="s",s=100,label="FSA",color="red",alpha=0.6)
plt.scatter(e_fsa,overlap_fsa,marker="s",s=100,label="FSA (CW/ACW)",color="red",alpha=0.6)
plt.scatter(e_gfsa,overlap_gfsa,marker="D",s=100,label="gFSA",color="green",alpha=0.6)
plt.scatter(e_krylov,overlap_krylov,marker="x",s=200,color="cyan",label="Krylov")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()

u_krylov_comp = np.dot(krylov_basis,u_krylov)
u_fsa_comp = np.dot(fsa_basis,u_fsa)
u_gfsa_comp = np.dot(gfsa_basis,u_gfsa)

exact_overlap_krylov = np.zeros(np.size(e_krylov))
exact_overlap_fsa = np.zeros(np.size(e_fsa))
exact_overlap_gfsa = np.zeros(np.size(e_gfsa))
for n in range(0,np.size(u_gfsa_comp,axis=1)):
    max_overlap_krylov = 0
    max_overlap_fsa = 0
    max_overlap_gfsa = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp_krylov = np.abs(np.vdot(u_krylov_comp[:,n],H.sector.eigvectors()[:,m]))
        temp_fsa = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))
        temp_gfsa = np.abs(np.vdot(u_gfsa_comp[:,n],H.sector.eigvectors()[:,m]))
        if temp_krylov> max_overlap_krylov:
            max_overlap_krylov = temp_krylov
        if temp_fsa> max_overlap_fsa:
            max_overlap_fsa = temp_fsa
        if temp_gfsa> max_overlap_gfsa:
            max_overlap_gfsa = temp_gfsa
    exact_overlap_krylov[n] = max_overlap_krylov
    exact_overlap_fsa[n] = max_overlap_fsa
    exact_overlap_gfsa[n] = max_overlap_gfsa

plt.plot(e_krylov,exact_overlap_krylov,marker="s",label="Krylov")
plt.plot(e_fsa,exact_overlap_fsa,marker="s",label="FSA (CW/ACW)")
# plt.plot(e_fsa,exact_overlap_fsa,marker="s",label="FSA")
plt.plot(e_gfsa,exact_overlap_gfsa,marker="D",alpha=0.6,label="gFSA")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
plt.legend()
plt.show()
# np.save("fsa,sub_raising,e,"+str(pxp.base)+","+str(pxp.N),e_gfsa)
# np.save("fsa,sub_raising,overlap,"+str(pxp.base)+","+str(pxp.N),overlap_gfsa)
# np.save("fsa,sub_raising,exact_overlap,"+str(pxp.base)+","+str(pxp.N),exact_overlap_gfsa)
        
    
