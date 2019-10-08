#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as scipy
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
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

def q_bracket_int(x,q):
    if q!=1:
        return (np.power(q,x)-np.power(q,-x))/(q-1/q)
    else:
        return x

def q_bracket(M,q):
    if q!=1:
        M_new = np.zeros((np.size(M,axis=0),np.size(M,axis=1)),dtype=complex)
        for n in range(0,np.size(M,axis=0)):
            for m in range(0,np.size(M,axis=0)):
                M_new[n,m] = (np.power(q,M[n,m])-np.power(q,-M[n,m]))/(q-1/q)
        return M_new
    else:
        return M

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

N = 10
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

    
#compare against pxp projected to fsa
pxp2 = unlocking_System([0],"periodic",2,N)
pxp2.gen_basis()

# a = 0.108

# a = 0.05
# H = Hamiltonian(pxp2)
# H.site_ops[1] = np.array([[0,1],[1,0]])
# H.model = np.array([[0,1,0],[0,0,1,0],[0,1,0,0]])
# H.model_coef = np.array([1,a,a])
# H.gen()

H = spin_Hamiltonian(pxp2,"x")
H.gen()

Hp = Hamiltonian(pxp2)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
# Hp.model = np.array([[0,1,0],[0,2,0],[0,0,1,0],[0,1,0,0],[0,2,0,0],[0,0,2,0]])
# Hp.model_coef = np.array([1,1,a,a,a,a])
# Hp.uc_size = np.array([2,2,2,2,2,2])
# Hp.uc_pos = np.array([1,0,0,1,0,1])
Hp.model = np.array([[0,1,0],[0,2,0]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([2,2])
Hp.uc_pos = np.array([1,0])
Hp.gen()

z=zm_state(2,1,pxp2,1)
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,pxp.N):
    next_state = np.dot(Hp.sector.matrix(),current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
e_fsa_diffs = np.zeros(np.size(e_fsa)-1)
for n in range(0,np.size(e_fsa_diffs,axis=0)):
    e_fsa_diffs[n] = e_fsa[n+1]-e_fsa[n]


e_central_gap = e_fsa_diffs[int(pxp.N/2)]

#find q such that central gap is same
def suq2_central_gap(q):
    sp = np.zeros((pxp.base,pxp.base),dtype=complex)
    s=1/2*(pxp.base-1)
    m = np.arange(-s,s)
    for n in range(0,np.size(m,axis=0)):
        coupling = np.power(q_bracket_int(s-m[n],q)*q_bracket_int(s+m[n]+1,q),0.5)
        sp[n+1,n] = coupling
    sm = np.conj(np.transpose(sp))
    m=np.arange(-s,s+1)
    sz = np.diag(m)
    raising_coef = sp[1,0]

    # # full chain operators
    Sp = np.zeros((pxp.dim,pxp.dim),dtype=complex)
    for n in range(0,np.size(pxp.basis,axis=0)):
        state_bits = np.copy(pxp.basis[n])
        for site in range(0,pxp.N):
            #need string of q^Sz LHS, q^-Sz RHS
            string = 1
            for m in range(0,site):
                z_eig = sz[state_bits[m],state_bits[m]]
                string = string * np.power(q,z_eig)
            for m in range(site+1,pxp.N):
                z_eig = sz[state_bits[m],state_bits[m]]
                string = string * np.power(q,-z_eig)

            coef = string * raising_coef #for spin 1 only

            site_ref = state_bits[site]
            new_bits = np.copy(state_bits)
            if site_ref == 0:
                new_bits[site] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    new_index = pxp.keys[new_ref]
                    Sp[new_index,n] = Sp[new_index,n] + coef
            # elif site_ref == 1:
                # new_bits[site] = 2
                # new_ref = bin_to_int_base_m(new_bits,pxp.base)
                # if new_ref in pxp.basis_refs:
                    # new_index = pxp.keys[new_ref]
                    # Sp[new_index,n] = Sp[new_index,n] + coef
    Sm = np.conj(np.transpose(Sp))

    H = (Sp + Sm)/2
    e,u = scipy.linalg.eigh(H)

    # for count in range(0,np.size(pxp.basis,axis=0)):
    z_energy = np.conj(u[0,:])
    overlap = np.log10(np.abs(z_energy)**2)
    # plt.scatter(e,overlap)
    # plt.show()
    eigenvalues = e

    to_del=[]
    for n in range(0,np.size(overlap,axis=0)):
        if overlap[n] <-5:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        overlap=np.delete(overlap,to_del[n])
        eigenvalues=np.delete(eigenvalues,to_del[n])

    e_diff = np.zeros(np.size(eigenvalues)-1)
    for n in range(0,np.size(eigenvalues,axis=0)-1):
        e_diff[n] = eigenvalues[n+1] - eigenvalues[n]
    to_del=[]
    for n in range(0,np.size(e_diff,axis=0)):
        if np.abs(e_diff[n]) <1e-5:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        e_diff=np.delete(e_diff,to_del[n])

    print(q,e_diff[int(pxp.N/2)])
    return e_diff[int(pxp.N/2)]


from scipy.optimize import minimize_scalar
# res = minimize_scalar(lambda q: np.abs(e_central_gap - suq2_central_gap(q)),method="golden",bracket=(0.7,1.3))
# res = minimize_scalar(lambda q: np.abs(1.33082727 - suq2_central_gap(q)),method="golden",bracket=(0.7,1.3))
# q=res.x
q=1.23374540
# q=1.248714324
# q=1.2
# q=1.3
print("q="+str(q))

sp = np.zeros((pxp.base,pxp.base),dtype=complex)
s=1/2*(pxp.base-1)
m = np.arange(-s,s)
for n in range(0,np.size(m,axis=0)):
    coupling = np.power(q_bracket_int(s-m[n],q)*q_bracket_int(s+m[n]+1,q),0.5)
    sp[n+1,n] = coupling
sm = np.conj(np.transpose(sp))
m=np.arange(-s,s+1)
sz = np.diag(m)
raising_coef = sp[1,0]

# # full chain operators
Sp = np.zeros((pxp.dim,pxp.dim),dtype=complex)
for n in range(0,np.size(pxp.basis,axis=0)):
    state_bits = np.copy(pxp.basis[n])
    for site in range(0,pxp.N):
        #need string of q^Sz LHS, q^-Sz RHS
        string = 1
        for m in range(0,site):
            z_eig = sz[state_bits[m],state_bits[m]]
            string = string * np.power(q,z_eig)
        for m in range(site+1,pxp.N):
            z_eig = sz[state_bits[m],state_bits[m]]
            string = string * np.power(q,-z_eig)

        coef = string * raising_coef #for spin 1 only

        site_ref = state_bits[site]
        new_bits = np.copy(state_bits)
        if site_ref == 0:
            new_bits[site] = 1
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            if new_ref in pxp.basis_refs:
                new_index = pxp.keys[new_ref]
                Sp[new_index,n] = Sp[new_index,n] + coef
        # elif site_ref == 1:
            # new_bits[site] = 2
            # new_ref = bin_to_int_base_m(new_bits,pxp.base)
            # if new_ref in pxp.basis_refs:
                # new_index = pxp.keys[new_ref]
                # Sp[new_index,n] = Sp[new_index,n] + coef
Sm = np.conj(np.transpose(Sp))

H = (Sp + Sm)/2
e,u = scipy.linalg.eigh(H)

# for count in range(0,np.size(pxp.basis,axis=0)):
z_energy = np.conj(u[0,:])
overlap = np.log10(np.abs(z_energy)**2)
# plt.scatter(e,overlap)
# plt.show()
eigenvalues = e

to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])

e_diff = np.zeros(np.size(eigenvalues)-1)
for n in range(0,np.size(eigenvalues,axis=0)-1):
    e_diff[n] = eigenvalues[n+1] - eigenvalues[n]
to_del=[]
for n in range(0,np.size(e_diff,axis=0)):
    if np.abs(e_diff[n]) <1e-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    e_diff=np.delete(e_diff,to_del[n])

print("Suq")
print(e_diff)
print("PXP FSA projected")
print(e_fsa_diffs)
print("Perm Projected")
temp = np.load("./E_diff_perm_N10.npy")
print(temp)

#compare time evolutions
t=np.arange(0,20,0.01)
f_fsa = np.zeros(np.size(t))
f_suq = np.zeros(np.size(t))

psi_fsa_energy = np.conj(u_fsa[0,:])
psi_suq_energy = np.conj(u[0,:])

for n in range(0,np.size(t,axis=0)):
    evolved_state_fsa = time_evolve_state(psi_fsa_energy,e_fsa,t[n])
    evolved_state_suq = time_evolve_state(psi_suq_energy,e,t[n])
    f_fsa[n] = np.abs(np.vdot(psi_fsa_energy,evolved_state_fsa))**2
    f_suq[n] = np.abs(np.vdot(psi_suq_energy,evolved_state_suq))**2
plt.plot(t,f_fsa,label = "PXP, FSA Projected")
plt.plot(t,f_suq,label = "Unconstrained SUq(2) Chain, q=1.2348")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$N=10$")
plt.legend()
plt.show()

plt.plot(e_diff,label="suq")
plt.plot(e_fsa_diffs,label="fsa")
plt.show()

overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)
overlap_suq = np.log10(np.abs(u[0,:])**2)
to_del=[]
for n in range(0,np.size(overlap_suq,axis=0)):
    if overlap_suq[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap_suq=np.delete(overlap_suq,to_del[n])
    e=np.delete(e,to_del[n])
    

plt.scatter(e,overlap_suq,label="fsa")
plt.scatter(e_fsa,overlap_fsa,marker="x",label="fsa")
plt.legend()

plt.legend()
plt.show()
# print(e_fsa)
# print(e)
