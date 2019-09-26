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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def get_top_band_indices(e,overlap):
    #identify top band to delete, want evolved dynamics just from second band
    #points closest to (200,200)
    d = np.zeros((np.size(overlap)))
    for n in range(0,np.size(overlap,axis=0)):
        if overlap[n] > -5:
            d[n] = np.power((e[n]-100)**2+(overlap[n]-150)**2,0.5)
        else:
            d[n] = 10000
    labels = np.arange(0,np.size(d))
    #N+1 largest vals
    d_sorted,labels_sorted = (list(t) for t in zip(*sorted(zip(d,labels))))
    scar_indices = labels_sorted[:int(N/2)]

    #points closest to (-200,200)
    d = np.zeros((np.size(overlap)))
    for n in range(0,np.size(overlap,axis=0)):
        if overlap[n] > -5:
            d[n] = np.power((e[n]+100)**2+(overlap[n]-150)**2,0.5)
        else:
            d[n] = 10000
    labels = np.arange(0,np.size(d))
    #N+1 largest vals
    d_sorted,labels_sorted = (list(t) for t in zip(*sorted(zip(d,labels))))
    scar_indices = np.append(scar_indices,labels_sorted[:int(N/2)])

    #identify zero energy state with largest overlap
    max_loc = None
    max_val = -1000
    for n in range(0,np.size(e,axis=0)):
        if np.abs(e[n])<1e-5:
            if overlap[n] > max_val:
                max_val = overlap[n]
                max_loc = n

    # if max_val > -1.5:
    scar_indices = np.append(scar_indices,max_loc)
    scar_indices = np.append(scar_indices,0)
    scar_indices = np.append(scar_indices,np.size(e,axis=0)-1)
    return scar_indices

N = 14
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0=spin_Hamiltonian(pxp,"x",pxp_syms)
V = Hamiltonian(pxp,pxp_syms)
V.site_ops[1] = np.array([[0,1],[1,0]])
V.site_ops[2] = np.array([[-1,0],[0,1]])
V.model = np.array([[0,1,1,1,0]])
V.model_coef = np.array([1])

# V.model = np.array([[0,1,0,2],[2,0,1,0]])
# V.model_coef = np.array([-1,-1])

z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

# for n in range(0,np.size(k,axis=0)):
    # H0.gen(k[n])
    # V.gen(k[n])

H0.gen()
V.gen()
H = H_operations.add(H0,V,np.array([1,0.1226121]))

#find rescaling such that E1-E0 = 1
def gap(H,c):
    e,u = np.linalg.eigh(c*H)
    gap = e[1]-e[0]
    print(gap,c)
    return gap

from scipy.optimize import minimize_scalar
# res = minimize_scalar(lambda c: np.abs(gap(H.sector.matrix(),c)-1),method="golden", bracket=(0.5,1.5))
# c=res.x
c=-0.8586044
H = H_operations.add(H0,V,np.array([c,c*0.1226121]))
H.sector.find_eig()

overlap = eig_overlap(z,H).eval()
scar_indices = np.unique(np.sort(get_top_band_indices(H.sector.eigvalues(),overlap)))

plt.scatter(H.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

scar_basis = np.zeros((pxp.dim,np.size(scar_indices)),dtype=complex)
for n in range(0,np.size(scar_indices,axis=0)):
    scar_basis[:,n] = H.sector.eigvectors()[:,scar_indices[n]]

Z_eff = np.zeros((np.size(scar_indices),np.size(scar_indices)))
for n in range(0,np.size(scar_indices,axis=0)):
    Z_eff[n,n] = H.sector.eigvalues()[scar_indices[n]]

for n in range(0,np.size(Z_eff,axis=0)):
    for m in range(0,np.size(Z_eff,axis=0)):
        temp = np.abs(Z_eff[n,m])
        if temp > 1e-5:
            print(temp,n,m)
    
S = 1/2*(np.size(Z_eff,axis=0)-1)

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
#pauli ops of su(2) subspace
m = np.arange(-S,S)
couplings = np.power(S*(S+1)-m*(m+1),0.5)
sP = np.diag(couplings,1)
sM = np.diag(couplings,-1)
Y = (sP - sM)/(2j)
X = (sP + sM)/(2)
Z = com(X,Y)

e,u = np.linalg.eigh(Y)
d=0.01
theta_vals = np.arange(0,2*math.pi+d,d)
F_vals = np.zeros(np.size(theta_vals))
for n in range(0,np.size(theta_vals,axis=0)):
# theta = math.pi/2
    theta = theta_vals[n]
    # D = np.diag(np.exp(1j * theta * e))
    # R = np.dot(np.conj(u),np.dot(D,u))
    R = sp.linalg.expm(-1j*theta*Y)

    X_eff = np.dot(np.transpose(np.conj(R)),np.dot(Z_eff,R))
    diff = X-X_eff
    F_norm = np.power(np.trace(np.dot(diff,np.conj(np.transpose(diff)))),0.5)
    F_vals[n] = F_norm
print(np.min(F_vals))
plt.plot(theta_vals,F_vals)
plt.axvline(x=math.pi/2)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\vert X-X_{eff} \vert_F$")
plt.title(r"Comparing exact pauli $X (2S+1 = 2N+1)$ rep, to approximate $X_{eff}$")
plt.show()

theta =math.pi/2
R = sp.linalg.expm(-1j*theta*Y)
X_eff = np.dot(np.transpose(np.conj(R)),np.dot(Z_eff,R))
plt.matshow(np.abs(X_eff))
plt.show()

print("X eff")
for n in range(0,np.size(X_eff,axis=0)):
    for m in range(0,np.size(X_eff,axis=0)):
        temp = np.abs(X_eff[n,m])
        if temp > 1e-5:
            print(temp,n,m)
