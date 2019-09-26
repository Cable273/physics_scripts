#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as scipy
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
from System_Classes import unlocking_System,U1_system
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

N = 8
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

# theta_vals = np.arange(0,2*math.pi,0.01)
# for count in range(0,np.size(theta_vals,axis=0)):
# q=np.exp(1j*2*math.pi/3)
q=1.23345
# site su(2) raising/z (q deformed)
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
plt.scatter(e,overlap)
plt.show()
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
    
print(e_diff)

t=np.arange(0,20,0.01)
evolved_state = np.zeros((pxp.dim,np.size(t,axis=0)),dtype=complex)
for n in range(0,np.size(t,axis=0)):
    evolved_state[:,n] = time_evolve_state(z_energy,e,t[n])

f=np.zeros((pxp.dim,np.size(t)))
pbar=ProgressBar()
for n in pbar(range(0,np.size(f,axis=0))):
    psi_energy = np.conj(u[n,:])
    for m in range(0,np.size(t,axis=0)):
        f[n,m] = np.abs(np.vdot(evolved_state[:,m],psi_energy))**2
for n in range(0,np.size(f,axis=0)):
    plt.plot(t,f[n,:])
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$SUq(2)$ Spin 1/2 chain, $H=S^+ + S^-$, $q=$"+str(q)+", $N=$"+str(N))
# plt.title(r"$SUq(2)$ Spin chain, $H=S^+ + S^-$, $q=e^{2i \pi/3}$, $N=$"+str(N))
# plt.title(r"$SUq(2)$ Spin chain, $H=S^+ + S^-$, $q=e^{i \theta}, \theta=0.01$, $N=$"+str(N))
plt.show()

f_init = f[0,:]
for n in range(0,np.size(f_init,axis=0)):
    if f_init[n]<0.1:
        cut = n
        break
max_index = np.argmax(f_init[cut:])
f_max = f_init[cut:][max_index]
t0 = t[cut:][max_index]
print(t0,f_max)
    
