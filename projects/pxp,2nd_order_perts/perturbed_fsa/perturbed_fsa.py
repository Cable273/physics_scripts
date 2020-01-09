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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
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

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,1,0],[0,2,0]])
Hp[0].model_coef = np.array([1,1])
Hp[0].uc_size = np.array([2,2])
Hp[0].uc_pos = np.array([1,0])

#1st order pert
Hp[1] = Hamiltonian(pxp)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,2,0],[0,2,0,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2])
Hp[1].uc_pos = np.array([0,1,1,0])

#2nd order perts
Hp[2] = Hamiltonian(pxp)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].model = np.array([[0,3,0,1,0],[0,1,0,3,0],[0,3,0,2,0],[0,2,0,3,0]])
Hp[2].model_coef = np.array([1,1,1,1])
Hp[2].uc_size = np.array([2,2,2,2])
Hp[2].uc_pos = np.array([1,1,0,0])

Hp[3] = Hamiltonian(pxp)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].model = np.array([[0,1,0,0,0],[0,0,0,1,0],[0,2,0,0,0],[0,0,0,2,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([2,2,2,2])
Hp[3].uc_pos = np.array([1,1,0,0])

Hp[4] = Hamiltonian(pxp)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].model = np.array([[0,0,1,0,0],[0,0,2,0,0]])
Hp[4].model_coef = np.array([1,1])
Hp[4].uc_size = np.array([2,2])
Hp[4].uc_pos = np.array([0,1])

Hp[5] = Hamiltonian(pxp)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].model = np.array([[0,0,1,0,3,0],[0,3,0,1,0,0],[0,0,2,0,3,0],[0,3,0,2,0,0]])
Hp[5].model_coef = np.array([1,1,1,1])
Hp[5].uc_size = np.array([2,2,2,2])
Hp[5].uc_pos = np.array([0,1,1,0])

Hp[6] = Hamiltonian(pxp)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].model = np.array([[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,2,0,0],[0,0,2,0,0,0]])
Hp[6].model_coef = np.array([1,1,1,1])
Hp[6].uc_size = np.array([2,2,2,2])
Hp[6].uc_pos = np.array([1,0,0,1])

Hp[7] = Hamiltonian(pxp)
Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[7].model = np.array([[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,3,0,2,0],[0,2,0,3,0,0]])
Hp[7].model_coef = np.array([1,1,1,1])
Hp[7].uc_size = np.array([2,2,2,2])
Hp[7].uc_pos = np.array([1,0,1,0])

Hp[8] = Hamiltonian(pxp)
Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[8].model = np.array([[0,0,0,0,1,0],[0,1,0,0,0,0],[0,0,0,0,2,0],[0,2,0,0,0,0]])
Hp[8].model_coef = np.array([1,1,1,1])
Hp[8].uc_size = np.array([2,2,2,2])
Hp[8].uc_pos = np.array([0,1,1,0])

Hp[9] = Hamiltonian(pxp)
Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[9].model = np.array([[0,0,1,0,3,0,0],[0,0,3,0,1,0,0],[0,0,2,0,3,0,0],[0,0,3,0,2,0,0]])
Hp[9].model_coef = np.array([1,1,1,1])
Hp[9].uc_size = np.array([2,2,2,2])
Hp[9].uc_pos = np.array([0,0,1,1])


for n in range(0,len(Hp)):
    Hp[n].gen()

# coef = np.array([0.0008,-1.43,0.0979,0.0980])
# coef = np.array([0.1897,0.2322,0.0312,0.0014,-0.0023,-0.00492,0.0056,0.0006,-0.0134,0.0130,-0.0138,-0.0253,-0.0623,0.0166,-0.0281,0.0200])
coef = np.array([0.11135,0.000217,-0.000287,-0.00717,0.00827,0.00336,0.00429,0.0103,0.00118])

from copy import deepcopy
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))

Hp = Hp_total.sector.matrix()
Hm = np.conj(np.transpose(Hp))

H = Hp+Hm

z=zm_state(2,1,pxp,1)
e,u = np.linalg.eigh(H)
overlap = np.log10(np.abs(u[pxp.keys[z.ref],:])**2)
psi_energy = np.conj(u[pxp.keys[z.ref],:])
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.show()

from Calculations import gen_fsa_basis
# fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),int(2*pxp.N/3))
fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),int(pxp.N))
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H,fsa_basis))


to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] < -10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    e=np.delete(e,to_del[n])
    
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)

plt.scatter(e,overlap)
plt.scatter(e_fsa,overlap_fsa,marker="x",color="red",s=100,label="FSA")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"$PXP, Z2$ Second Order SU(2) Perts, FSA, N="+str(pxp.N))
plt.legend()
plt.show()

exact_overlap = np.zeros(np.size(fsa_basis,axis=1))
u_fsa_comp = np.dot(fsa_basis,u_fsa)
for n in range(0,np.size(u_fsa_comp,axis=1)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_fsa_comp[:,n],u[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.plot(e_fsa,exact_overlap,marker="s")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"$PXP, Z2$ Second Order SU(2) Perts, FSA, N="+str(pxp.N))
print(exact_overlap)
plt.show()
