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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT,inversion
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
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=20
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=4),PT(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,1,2,1,0],[0,2,1,2,0],[0,2,1,2,0],[0,1,2,1,0]])
Hp[0].model_coef = np.array([1,1,1,1])
Hp[0].uc_size = np.array([4,4,4,4])
Hp[0].uc_pos = np.array([2,3,0,1])

# Hp[1] = Hamiltonian(pxp,pxp_syms)
# Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[1].model = np.array([[0,2,1,0,4,0,0],[0,0,4,0,1,2,1,0],[0,0,4,0,2,1,2,0],[0,1,2,1,0,4,0,0]])
# Hp[1].model_coef = np.array([1,1,1,1])
# Hp[1].uc_size = np.array([4,4,4,4])
# Hp[1].uc_pos = np.array([0,2,0,2])
# Hp[1].gen()

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
Hp[1].model = np.array([[0,2,1,2,0,4,0,0],[0,0,4,0,1,2,1,0],[0,0,4,0,2,1,2,0],[0,1,2,1,0,4,0,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([4,4,4,4])
Hp[1].uc_pos = np.array([3,3,1,1])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[0,2,1,2,0,4,0],[0,2,1,2,0,4,0],[0,4,0,2,1,2,0],[0,4,0,2,1,2,0],[0,1,2,1,0,4,0],[0,1,2,1,0,4,0],[0,4,0,1,2,1,0],[0,4,0,1,2,1,0]])
Hp[2].model_coef = np.array([1,1,1,1,1,1,1,1])
Hp[2].uc_size = np.array([4,4,4,4,4,4,4,4,])
Hp[2].uc_pos = np.array([0,3,2,1,1,2,3,0])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[4] = np.array([[0,0],[0,1]])
Hp[3].model = np.array([[0,1,2,0,4,0,2,0],[0,1,0,4,0,1,2,0],[0,2,1,0,4,0,1,0],[0,2,0,4,0,2,1,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([4,4,4,4])
Hp[3].uc_pos = np.array([2,0,0,2])


z=zm_state(4,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,len(Hp)):
    for m in range(0,np.size(k,axis=0)):
        Hp[n].gen(k[m])

# for m in range(0,np.size(k,axis=0)):
    # Hp[0].gen(k[m])

coef = np.load("./pxxxp,1stOrder,coef,12.npy")
# coef = np.zeros(len(Hp)-1)
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hm = Hp_total.herm_conj()

H = H_operations.add(Hp_total,Hm,np.array([1,1]))
H.sector.find_eig(k[0])
eig_overlap(z,H,k[0]).plot(tol=-15)
plt.show()
fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
plt.show()

from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(k[0]),z.sym_basis(k[0],pxp_syms),int(2*pxp.N/4))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(k[0]),fsa_basis))
e,u = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(u[0,:])**2)

exact_overlap = eig_overlap(z,H,k[0]).eval()
plt.scatter(H.sector.eigvalues(k[0]),exact_overlap)
plt.scatter(e,overlap_fsa,marker="x",color="red",s=100)
plt.show()
t=np.arange(0,20,0.01)
f = fidelity(z,H,"use sym").eval(t,z)
plt.plot(t,f)
plt.show()

u_comp = np.dot(fsa_basis,u)
fsa_exact_overlap = np.zeros(np.size(u_comp,axis=1))
for n in range(0,np.size(fsa_exact_overlap,axis=0)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvectors(k[0]),axis=1)):
        temp = np.abs(np.vdot(H.sector.eigvectors(k[0])[:,m],u_comp[:,n]))**2
        if temp > max_overlap:
            max_overlap = temp
    fsa_exact_overlap[n] = max_overlap
plt.plot(e,fsa_exact_overlap,marker="s")
plt.show()

exact_energy = H.sector.eigvalues(k[0])
fsa_energy = e
fsa_overlap = overlap_fsa
        
# np.save("pxxxp,0th_order,e,"+str(pxp.N),exact_energy)
# np.save("pxxxp,0th_order,LW_overlap,"+str(pxp.N),exact_overlap)
# np.save("pxxxp,0th_order,LW_fidelity,"+str(pxp.N),f)
# np.save("pxxxp,LW_fsa,0th_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxxxp,LW_fsa,0th_order,LW_overlap,"+str(pxp.N),fsa_overlap)
# np.save("pxxxp,LW_fsa,0th_order,fsa_exact_overlap,"+str(pxp.N),fsa_exact_overlap)

np.save("pxxxp,1st_order,e,"+str(pxp.N),exact_energy)
np.save("pxxxp,1st_order,LW_overlap,"+str(pxp.N),exact_overlap)
np.save("pxxxp,1st_order,LW_fidelity,"+str(pxp.N),f)
np.save("pxxxp,LW_fsa,1st_order,e,"+str(pxp.N),fsa_energy)
np.save("pxxxp,LW_fsa,1st_order,LW_overlap,"+str(pxp.N),fsa_overlap)
np.save("pxxxp,LW_fsa,1st_order,fsa_exact_overlap,"+str(pxp.N),fsa_exact_overlap)


