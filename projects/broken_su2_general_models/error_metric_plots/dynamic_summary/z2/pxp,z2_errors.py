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
from copy import deepcopy

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
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
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=20
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,1,0],[0,2,0]])
Hp[0].model_coef = np.array([1,1])
Hp[0].uc_size = np.array([2,2])
Hp[0].uc_pos = np.array([1,0])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,2,0],[0,2,0,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2])
Hp[1].uc_pos = np.array([0,1,1,0])

#2nd order perts
Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].model = np.array([[0,3,0,1,0],[0,1,0,3,0],[0,3,0,2,0],[0,2,0,3,0]])
Hp[2].model_coef = np.array([1,1,1,1])
Hp[2].uc_size = np.array([2,2,2,2])
Hp[2].uc_pos = np.array([1,1,0,0])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].model = np.array([[0,1,0,0,0],[0,0,0,1,0],[0,2,0,0,0],[0,0,0,2,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([2,2,2,2])
Hp[3].uc_pos = np.array([1,1,0,0])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].model = np.array([[0,0,1,0,0],[0,0,2,0,0]])
Hp[4].model_coef = np.array([1,1])
Hp[4].uc_size = np.array([2,2])
Hp[4].uc_pos = np.array([0,1])

Hp[5] = Hamiltonian(pxp,pxp_syms)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].model = np.array([[0,0,1,0,3,0],[0,3,0,1,0,0],[0,0,2,0,3,0],[0,3,0,2,0,0]])
Hp[5].model_coef = np.array([1,1,1,1])
Hp[5].uc_size = np.array([2,2,2,2])
Hp[5].uc_pos = np.array([0,1,1,0])

Hp[6] = Hamiltonian(pxp,pxp_syms)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].model = np.array([[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,2,0,0],[0,0,2,0,0,0]])
Hp[6].model_coef = np.array([1,1,1,1])
Hp[6].uc_size = np.array([2,2,2,2])
Hp[6].uc_pos = np.array([1,0,0,1])

Hp[7] = Hamiltonian(pxp,pxp_syms)
Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[7].model = np.array([[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,3,0,2,0],[0,2,0,3,0,0]])
Hp[7].model_coef = np.array([1,1,1,1])
Hp[7].uc_size = np.array([2,2,2,2])
Hp[7].uc_pos = np.array([1,0,1,0])

Hp[8] = Hamiltonian(pxp,pxp_syms)
Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[8].model = np.array([[0,0,0,0,1,0],[0,1,0,0,0,0],[0,0,0,0,2,0],[0,2,0,0,0,0]])
Hp[8].model_coef = np.array([1,1,1,1])
Hp[8].uc_size = np.array([2,2,2,2])
Hp[8].uc_pos = np.array([0,1,1,0])

Hp[9] = Hamiltonian(pxp,pxp_syms)
Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[9].model = np.array([[0,0,1,0,3,0,0],[0,0,3,0,1,0,0],[0,0,2,0,3,0,0],[0,0,3,0,2,0,0]])
Hp[9].model_coef = np.array([1,1,1,1])
Hp[9].uc_size = np.array([2,2,2,2])
Hp[9].uc_pos = np.array([0,0,1,1])

z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,len(Hp)):
    # Hp[n].gen()
    for m in range(0,np.size(k,axis=0)):
        Hp[n].gen(k[m])

# coef = np.zeros(9)
coef = np.load("../../../../pxp,2nd_order_perts/z2/data/all_terms/18/18_all_terms/pxp,z2,2nd_order_perts,fid_coef,18.npy")
# coef[0] = 0.108
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hm = Hp_total.herm_conj()

H = H_operations.add(Hp_total,Hm,np.array([1,1]))
z=zm_state(2,1,pxp,1)
fsa_dim = pxp.N
from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(k[0]),z.sym_basis(k[0],pxp_syms),fsa_dim)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(k[0]),fsa_basis))

H.sector.find_eig(k[0])
exact_overlap = eig_overlap(z,H,k[0]).eval()
exact_energy = H.sector.eigvalues(k[0])

e,u = np.linalg.eigh(H_fsa)
fsa_overlap = np.log10(np.abs(u[0,:])**2)
fsa_energy = e

plt.scatter(exact_energy,exact_overlap)
plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
plt.show()

t=np.arange(0,20,0.01)
f=fidelity(z,H,"use sym").eval(t,z)
plt.plot(t,f)
plt.show()

#identify scar states for entropy highlight
scar_indices = get_top_band_indices(exact_energy,exact_overlap,pxp.N,100,100,e_diff = 0.5)
#check identified right states
plt.scatter(exact_energy,exact_overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(exact_energy[scar_indices[n]],exact_overlap[scar_indices[n]],marker="D",color="green",alpha=0.5,s=100)
plt.show()

U = pxp_syms.basis_transformation(k[0])
eigvectors_comp = np.dot(U,H.sector.eigvectors(k[0]))
fsa_eigs_comp = np.dot(U,np.dot(fsa_basis,u))

#entropy
ent_vals = np.zeros(np.size(eigvectors_comp,axis=1))
ent = entropy(pxp)
pbar=ProgressBar()
for n in pbar(range(0,np.size(ent_vals,axis=0))):
    ent_vals[n] = ent.eval(eigvectors_comp[:,n])

ent_fsa = np.zeros(np.size(e))
for n in range(0,np.size(ent_fsa,axis=0)):
    ent_fsa[n] = ent.eval(fsa_eigs_comp[:,n])

plt.scatter(exact_energy,ent_vals)
scar_entropy = np.zeros(np.size(scar_indices))
scar_energy = np.zeros(np.size(scar_indices))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_entropy[n] = ent_vals[scar_indices[n]]
    scar_energy[n] = exact_energy[scar_indices[n]]
plt.scatter(scar_energy,scar_entropy,marker="D",color="orange",alpha=0.4,s=200,label="ED Scars")
plt.scatter(e,ent_fsa,marker="x",color="red",s=200,label=r"$su(2)$ Ritz vectors")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.show()

# np.save("pxp,0th_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,0th_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,0th_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,0th_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,0th_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pxp,1st_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,1st_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,1st_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,1st_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,1st_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pxp,2nd_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,2nd_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxp,2nd_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pxp,z2_fsa,2nd_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z2_fsa,2nd_order,z2_overlap,"+str(pxp.N),fsa_overlap)
