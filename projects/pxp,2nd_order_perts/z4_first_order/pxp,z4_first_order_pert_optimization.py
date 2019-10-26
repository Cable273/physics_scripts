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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
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

N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,2,0],[0,1,0],[0,1,0],[0,1,0]])
Hp[0].model_coef = np.array([1,1,1,1])
Hp[0].uc_size = np.array([4,4,4,4])
Hp[0].uc_pos = np.array([3,0,1,2])

Hp[1] = Hamiltonian(pxp)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,1,2,1,0]])
Hp[1].model_coef = np.array([1])
Hp[1].uc_size = np.array([4])
Hp[1].uc_pos = np.array([0])

Hp[2] = Hamiltonian(pxp)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[2].model_coef = np.array([1,1])
Hp[2].uc_size = np.array([4,4])
Hp[2].uc_pos = np.array([3,1])

Hp[3] = Hamiltonian(pxp)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([4,4,4,4])
Hp[3].uc_pos = np.array([0,0,1,1])

Hp[4] = Hamiltonian(pxp)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Hp[4].model_coef = np.array([1,1,1,1])
Hp[4].uc_size = np.array([4,4,4,4])
Hp[4].uc_pos = np.array([2,3,3,2])

for n in range(0,len(Hp)):
    Hp[n].gen()

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

from copy import deepcopy
from Hamiltonian_Classes import H_operations
from scipy.optimize import minimize,minimize_scalar
def fidelity_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    e,u = np.linalg.eigh(H)
    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

    t=np.arange(0,10,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        f[n] = -fidelity_eval(psi_energy,e,t[n])
    # plt.plot(t,f)
    # plt.show()
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f_max = np.max(f[cut:])
        
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(4.5,5.5))
    f = -fidelity_eval(psi_energy,e,res.x)
    print(coef,f)
    # print(f)
    if res.x <1e-5:
        return 1000
    else:
        return -f_max
    # if (np.abs(coef)>0.5).any():
        # return 1000
    # return -f

from Calculations import get_top_band_indices
def spacing_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    e,u = np.linalg.eigh(H)
    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())
    overlap = np.log10(np.abs(psi_energy)**2)
    scar_indices = get_top_band_indices(e,overlap,int(2*N/3),150,200,e_diff=0.5)
    # plt.scatter(e,overlap)
    # for n in range(0,np.size(scar_indices,axis=0)):
        # plt.scatter(e[scar_indices[n]],overlap[scar_indices[n]],marker="x",s=100,color="red")
    # plt.show()

    scar_e = np.zeros(np.size(scar_indices))
    for n in range(0,np.size(scar_indices,axis=0)):
        scar_e[n] = e[scar_indices[n]]
    diffs = np.zeros(np.size(scar_e)-1)
    for n in range(0,np.size(diffs,axis=0)):
        diffs[n] = scar_e[n+1] - scar_e[n]
    diff_matrix = np.zeros((np.size(diffs),np.size(diffs)))
    for n in range(0,np.size(diff_matrix,axis=0)):
        for m in range(0,np.size(diff_matrix,axis=0)):
            diff_matrix[n,m] = np.abs(diffs[n] - diffs[m])
    error = np.power(np.trace(np.dot(diff_matrix,np.conj(np.transpose(diff_matrix)))),0.5)
    print(coef,error)
    print(scar_e)
    # print(coef)
    # if (np.abs(coef).any())>0.5:
        # return 1000
    # else:
    return error

# coef = np.ones(4)*0.1
# coef = np.array([0.00152401,-1.42847732,0.09801341,0.09784458])
# res = minimize(lambda coef:fidelity_error(coef),method="powell",x0=coef)
# coef = res.x
coef = np.load("./data/pxp,z4,pert_coef.npy")
# coef = np.zeros(4)

print(coef)
error = spacing_error(coef)
print("ERROR VALUE: "+str(error))

print(coef)
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hp = Hp_total.sector.matrix()
Hm = np.conj(np.transpose(Hp))

H = Hp+Hm
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 1/2*com(Hp,Hm)

e,u = np.linalg.eigh(H)
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
z=zm_state(4,1,pxp)
psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(3.5,5.5))
f0 = fidelity_eval(psi_energy,e,res.x)
print("F0 REVIVAL")
print(res.x,fidelity_eval(psi_energy,e,res.x))
# np.savetxt("pxp,z4,f0,"+str(pxp.N),[f0])

z2=zm_state(4,1,pxp,1)
z3=zm_state(4,1,pxp,2)
z4=zm_state(4,1,pxp,3)
for n in range(0,np.size(t,axis=0)):
    f[n] = -fidelity_eval(psi_energy,e,t[n])
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$PXP, Z_4$, first order pertubations, N="+str(pxp.N))
plt.show()

overlap = np.log10(np.abs(psi_energy)**2)
eigenvalues = np.copy(e)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
    
plt.scatter(eigenvalues,overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_4 \vert E \rangle \vert^2)$")
plt.title(r"$PXP, Z_4$, first order pertubations, N="+str(pxp.N))
plt.show()

#check harmonic spacing
def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

z=zm_state(4,1,pxp)

fsa_basis = z.prod_basis()
current_state = fsa_basis
fsa_dim = int(2*(pxp.N/4))
for n in range(0,fsa_dim):
    next_state = norm(np.dot(Hp,current_state))
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
    
Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
for n in range(0,np.size(fsa_basis,axis=1)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])
    print(Hz_exp[n],Hz_var[n])

# print("\n")
# e_diff = np.zeros(np.size(Hz_exp)-1)
# for n in range(0,np.size(e_diff,axis=0)):
    # e_diff[n] = Hz_exp[n+1]-Hz_exp[n]
    # print(e_diff[n])
    
# np.save("pxp,z4,pert_coef",coef)
# np.savetxt("pxp,z4,Hz_eigvalues,"+str(pxp.N),e)
# np.savetxt("pxp,z4,Hz_var,"+str(pxp.N),var_vals)
# np.savetxt("pxp,z4,Hz_eig_diffs,"+str(pxp.N),e_diff)
# # np.savetxt("pxp,z4,f0,"+str(pxp.N),[f0])
# np.savetxt("pxp,z4,spacing_error,"+str(pxp.N),[error])
