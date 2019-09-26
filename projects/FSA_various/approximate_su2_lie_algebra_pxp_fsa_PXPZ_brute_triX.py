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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,get_top_band_indices

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

# pert_coef = 0.1226
pert_coef = 0.051
# H0=spin_Hamiltonian(pxp,"x",pxp_syms)

# V = Hamiltonian(pxp,pxp_syms)
# V.site_ops[1] = np.array([[0,1],[1,0]])
# V.site_ops[2] = np.array([[-1,0],[0,1]])

# V.model = np.array([[2,0,1,0],[0,1,0,2]])
# V.model_coef = np.array([1,1])

# # V.model = np.array([[0,1,1,1,0]])
# # V.model_coef = np.array([1])

# V.gen()
# H0.gen()

# H=H_operations.add(H0,V,np.array([1,-pert_coef]))

H=Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[-1,0],[0,1]])
H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0],[2,-1,0,1,0],[0,1,0,-1,2],[2,-1,-1,0,1,0],[0,1,0,-1,-1,2],[2,-1,-1,-1,0,1,0],[0,1,0,-1,-1,-1,2]])
H.model_coef = np.array([1,-0.051,-0.051,-0.0102,-0.0102,-3.1875e-3,-3.1875e-3,-1.1333e-3,-1.1333e-3])

H.gen()
H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
plt.show()
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()

# find rescaling such that E1-E0 = 1
def gap(H,c):
    e,u = np.linalg.eigh(c*H)
    gap = e[1]-e[0]
    return np.abs(1-gap)

#set energy differences to 1
# from scipy.optimize import minimize_scalar
# res = minimize_scalar(lambda c: gap(H.sector.matrix(),c),method="golden",bracket=(0.7,0.8))
# c=res.x
# print(c)
# c=0.7832296641106475
c=0.77345834733

H=Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[-1,0],[0,1]])
H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0],[2,-1,0,1,0],[0,1,0,-1,2],[2,-1,-1,0,1,0],[0,1,0,-1,-1,2],[2,-1,-1,-1,0,1,0],[0,1,0,-1,-1,-1,2]])
H.model_coef = np.array([c,-c*0.051,-c*0.051,-c*0.0102,-c*0.0102,-c*3.1875e-3,-c*3.1875e-3,-c*1.1333e-3,-c*1.1333e-3])
H.gen()

# H=H_operations.add(H0,V,np.array([c,-c*pert_coef]))
H.sector.find_eig()
print("EIG DIFF")
print(H.sector.eigvalues()[1]-H.sector.eigvalues()[0])
z=zm_state(2,1,pxp)
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()
overlap = eig_overlap(z,H).eval()
scar_indices = np.unique(np.sort(get_top_band_indices(H.sector.eigvalues(),overlap,N)))

plt.scatter(H.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

scar_basis = np.zeros((pxp.dim,np.size(scar_indices)))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_basis[scar_indices[n],n] = 1

D=np.diag(H.sector.eigvalues())
Z_eff = np.dot(np.conj(np.transpose(scar_basis)),np.dot(D,scar_basis))

def X_cost(a):
    diagonal = np.append(a,np.flip(a))
    X_eff = np.diag(diagonal,1)+np.diag(diagonal,-1)

    dim = np.size(X_eff,axis=0)

    eigs = np.diag(Z_eff)
    v=np.zeros(dim)
    for n in range(0,dim):
        v[n] = np.linalg.det(eigs[n]*np.eye(dim)-X_eff)
    # print("Cost="+str(np.vdot(v,v)))
    return np.vdot(v,v)
        
from scipy.optimize import minimize
#try guess of 
red_dim = np.size(Z_eff,axis=0)
S=1/2*(red_dim-1)
m=np.arange(-S,S)
couplings = 1/2*np.power(S*(S+1)-m*(m+1),0.5)

init_guess=couplings[0:int(1/2*(np.size(Z_eff,axis=0)-1))]
# res = minimize(lambda a: X_cost(a),method="powell",x0=np.zeros(int(1/2*(np.size(Z_eff,axis=0)-1))))
res = minimize(lambda a: X_cost(a),method="powell",x0=init_guess)
diagonal = np.append(res.x,np.flip(res.x))
X_eff = np.diag(diagonal,1)+np.diag(diagonal,-1)

plt.matshow(np.abs(X_eff))
plt.show()
plt.plot(np.diag(X_eff,1))
plt.show()
e0,u0 = np.linalg.eigh(X_eff)

print("\n")
print("Z eff")
print(np.diag(Z_eff))
print("\n")

print("X eff")
print(e0)

Ux_eig = u0
P = scar_basis
U_eig = H.sector.eigvectors()

U_fsa = np.dot(P,np.conj(np.transpose(Ux_eig)))
U_fsa = np.dot(U_eig,U_fsa)
print(np.shape(U_fsa))
from Diagnostics import print_wf
print_wf(U_fsa[:,0],pxp,1e-2)

z=zm_state(2,1,pxp)
z_energy = H.sector.eigvectors()[pxp.keys[z.ref],:]

z0=U_fsa[:,2]
z0_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),z0)

overlap = np.log10(np.abs(z0_energy)**2)
overlap_neel = np.log10(np.abs(z_energy)**2)

plt.scatter(H.sector.eigvalues(),overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1} (\sum_{d=2}^5 Z_{i+d} + Z_{i-d} ), N=$"+str(N)+"\n Numerical state overlap")
plt.show()
plt.scatter(H.sector.eigvalues(),overlap_neel)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1} (\sum_{d=2}^5 Z_{i+d} + Z_{i-d} ), N=$"+str(N)+"\n Neel state overlap")
plt.show()

t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
f_neel=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z0_energy,H.sector.eigvalues(),t[n])
    evolved_state_neel = time_evolve_state(z_energy,H.sector.eigvalues(),t[n])
    f[n] = np.abs(np.vdot(z0_energy,evolved_state))**2
    f_neel[n] = np.abs(np.vdot(z_energy,evolved_state_neel))**2
plt.plot(t,np.log10(1-f),label="Numerical State")
plt.plot(t,np.log10(1-f_neel),label="Neel State")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\log(1-\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2)$")
plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1} (\sum_{d=2}^5 Z_{i+d} + Z_{i-d} ), N=$"+str(N))
plt.show()

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()
H0.sector.find_eig()

z0_energy = np.dot(np.conj(np.transpose(H0.sector.eigvectors())),z0)
z_energy = H0.sector.eigvectors()[pxp.keys[z.ref],:]

# overlap = np.log10(np.abs(z0_energy)**2)
# overlap_neel = np.log10(np.abs(z_energy)**2)

# plt.scatter(H.sector.eigvalues(),overlap)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1}, N=$"+str(N))
# plt.show()
# plt.scatter(H.sector.eigvalues(),overlap_neel)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1}, N=$"+str(N))
# plt.show()

t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
f_neel=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z0_energy,H0.sector.eigvalues(),t[n])
    evolved_state_neel = time_evolve_state(z_energy,H0.sector.eigvalues(),t[n])
    f[n] = np.abs(np.vdot(z0_energy,evolved_state))**2
    f_neel[n] = np.abs(np.vdot(z_energy,evolved_state_neel))**2
plt.plot(t,f,label="Numerical State")
plt.plot(t,f_neel,label="Neel State")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$\sum_{n} P_{n-1}X_nP_{n+1}, N=$"+str(N))
plt.show()


temp = np.dot(np.conj(np.transpose(U_fsa)),np.dot(H.sector.matrix(),U_fsa))
plt.matshow(np.abs(temp))
# print(np.diag(temp,1))
# print(np.diag(X_eff,1))

print("\n")
z_prod_basis = zm_state(2,1,pxp).prod_basis()
from Diagnostics import print_wf
for n in range(0,np.size(U_fsa,axis=1)):
    # print("\n")
    # print_wf(U_fsa[:,n],pxp,1e-2)
    overlap = np.abs(np.vdot(z_prod_basis,U_fsa[:,n]))
    # overlap = np.abs(np.vdot(z_prod_basis,fsa_basis[:,n]))
    # print(np.vdot(U_fsa[:,n],U_fsa[:,n]))
    print(100*overlap)
