#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
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

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp_rho(Q,rho):
    return np.trace(np.dot(rho,Q))
def var_rho(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp_rho(Q2,psi)-exp_rho(Q,psi)**2

def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))

def check_basis_config(basis_config,uc_size,allowed_occ):
    N = np.size(basis_config)
    pass_checks = 1
    for n in range(0,N):
        cell_occ = np.zeros(uc_size)
        for m in range(0,uc_size):
            cell_occ[m] = basis_config[int((n+m) % N)]
        if np.sum(cell_occ) > allowed_occ:
            pass_checks = 0
            break
    if pass_checks == 1:
        return True
    else:
        return False

# init system
N=18
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()

uc_size = 3
allowed_occ = 2

new_basis = np.zeros(pxp.N)
pbar=ProgressBar()
print("Decimating basis")
for n in pbar(range(0,np.size(pxp.basis,axis=0))):
    if check_basis_config(pxp.basis[n],uc_size,allowed_occ) == True:
        new_basis = np.vstack((new_basis,pxp.basis[n]))

pxp.basis = new_basis
pxp.basis_refs = np.zeros(np.size(pxp.basis,axis=0))
pxp.keys = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    pxp.basis_refs[n] = bin_to_int_base_m(pxp.basis[n],pxp.base)
    pxp.keys[pxp.basis_refs[n]] = n
pxp.dim = np.size(pxp.basis,axis=0)
print("new dim = " +str(pxp.dim))

pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

H = spin_Hamiltonian(pxp,"x",pxp_syms)
overlapSectors = dict()
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    overlapSectors[n] = eig_overlap(z,H,k[n]).eval()

overlap = np.append(overlapSectors[0],overlapSectors[1])
energy = np.append(H.sector.eigvalues(k[0]),H.sector.eigvalues(k[1]))

plt.scatter(energy,overlap)
from Calculations import get_top_band_indices
scar_indices = get_top_band_indices(energy,overlap,N,100,200,e_diff=0.5)
scar_e = np.zeros(np.size(scar_indices))
scar_overlap = np.zeros(np.size(scar_indices))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_e[n] = energy[scar_indices[n]]
    scar_overlap[n] = overlap[scar_indices[n]]
plt.scatter(scar_e,scar_overlap,marker="x",color="red",s=300)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
plt.show()

t=np.arange(0,20,0.01)
f = fidelity(z,H,"use sym").eval(t,z)
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.show()

U = dict()
for n in range(0,np.size(k,axis=0)):
    U[n] = pxp_syms.basis_transformation(k[n])

eigvectors = np.dot(U[0],H.sector.eigvectors(k[0]))
eigvectors = np.hstack((eigvectors,np.dot(U[1],H.sector.eigvectors(k[1]))))
ent_vals = np.zeros(np.size(eigvectors,axis=1))
ent = entropy(pxp)
pbar=ProgressBar()
for n in pbar(range(0,np.size(ent_vals))):
    ent_vals[n] = ent.eval(eigvectors[:,n])
scar_ent = np.zeros(np.size(scar_indices))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_ent[n] = ent_vals[scar_indices[n]]
    
plt.scatter(energy,ent_vals)
plt.scatter(scar_e,scar_ent,marker="x",color="red",s=300,label="Z2 Scars")
plt.xlabel(r"$E$")
plt.xlabel(r"$S$")
plt.legend()
plt.show()

# np.save("constrainedPara,3,2,eigvalues,"+str(pxp.N),energy)
# np.save("constrainedPara,3,2,overlap,"+str(pxp.N),overlap)
# np.save("constrainedPara,3,2,entropy,"+str(pxp.N),ent_vals)
# np.save("constrainedPara,3,2,t,"+str(pxp.N),t)
# np.save("constrainedPara,3,2,f,"+str(pxp.N),f)
# np.save("constrainedPara,3,2,scar_energy,"+str(pxp.N),scar_e)
# np.save("constrainedPara,3,2,scar_overlap,"+str(pxp.N),scar_overlap)
# np.save("constrainedPara,3,2,scar_entropy,"+str(pxp.N),scar_ent)

