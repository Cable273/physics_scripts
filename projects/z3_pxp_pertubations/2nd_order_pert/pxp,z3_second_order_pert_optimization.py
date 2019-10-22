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

N=15
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,2,0],[0,1,0],[0,1,0]])
Hp[0].model_coef = np.array([1,1,1])
Hp[0].uc_size = np.array([3,3,3])
Hp[0].uc_pos = np.array([2,0,1])

Hp[1] = Hamiltonian(pxp)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,2,0,0],[0,0,2,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([3,3,3,3])
Hp[1].uc_pos = np.array([2,1,2,1])

Hp[2] = Hamiltonian(pxp)
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].model = np.array([[0,0,2,0,0]])
Hp[2].model_coef = np.array([1])
Hp[2].uc_size = np.array([3])
Hp[2].uc_pos = np.array([1])

# Hp[3] = Hamiltonian(pxp)
# Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[3].model = np.array([[0,2,0,3,0],[0,3,0,2,0]])
# Hp[3].model_coef = np.array([1,1])
# Hp[3].uc_size = np.array([3,3])
# Hp[3].uc_pos = np.array([2,0])

# Hp[4] = Hamiltonian(pxp)
# Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[4].model = np.array([[0,2,0,3,0,0],[0,0,3,0,2,0]])
# Hp[4].model_coef = np.array([1,1])
# Hp[4].uc_size = np.array([3,3])
# Hp[4].uc_pos = np.array([2,2])

# Hp[5] = Hamiltonian(pxp)
# Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[5].model = np.array([[0,1,0,0],[0,0,1,0]])
# Hp[5].model_coef = np.array([1,1])
# Hp[5].uc_size = np.array([3,3])
# Hp[5].uc_pos = np.array([0,0])

# Hp[6] = Hamiltonian(pxp)
# Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[6].model = np.array([[0,0,1,0,0],[0,0,1,0,0]])
# Hp[6].model_coef = np.array([1,1])
# Hp[6].uc_size = np.array([3,3])
# Hp[6].uc_pos = np.array([2,0])

# Hp[7] = Hamiltonian(pxp)
# Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[7].model = np.array([[0,1,0,0,0],[0,0,0,1,0]])
# Hp[7].model_coef = np.array([1,1])
# Hp[7].uc_size = np.array([3,3])
# Hp[7].uc_pos = np.array([0,0])

# Hp[8] = Hamiltonian(pxp)
# Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[8].site_ops[3] = np.array([[-1/2,0],[1/2,0]])
# Hp[8].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
# Hp[8].model_coef = np.array([1,1])
# Hp[8].uc_size = np.array([3,3])
# Hp[8].uc_pos = np.array([1,1])

# Hp[9] = Hamiltonian(pxp)
# Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[9].model = np.array([[0,1,0,3,0],[0,3,0,1,0]])
# Hp[9].model_coef = np.array([1,1])
# Hp[9].uc_size = np.array([3,3])
# Hp[9].uc_pos = np.array([0,2])

# Hp[10] = Hamiltonian(pxp)
# Hp[10].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[10].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[10].model = np.array([[0,1,0,4,0],[0,4,0,1,0]])
# Hp[10].model_coef = np.array([1,1])
# Hp[10].uc_size = np.array([3,3])
# Hp[10].uc_pos = np.array([0,2])

# Hp[11] = Hamiltonian(pxp)
# Hp[11].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[11].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0]])
# Hp[11].model_coef = np.array([1,1])
# Hp[11].uc_size = np.array([3,3])
# Hp[11].uc_pos = np.array([2,2])

# Hp[12] = Hamiltonian(pxp)
# Hp[12].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[12].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[12].model = np.array([[0,0,1,0,3,0],[0,3,0,1,0,0]])
# Hp[12].model_coef = np.array([1,1])
# Hp[12].uc_size = np.array([3,3])
# Hp[12].uc_pos = np.array([2,2])

# Hp[13] = Hamiltonian(pxp)
# Hp[13].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[13].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[13].model = np.array([[0,1,0,3,0,0],[0,0,3,0,1,0]])
# Hp[13].model_coef = np.array([1,1])
# Hp[13].uc_size = np.array([3,3])
# Hp[13].uc_pos = np.array([0,1])

# Hp[14] = Hamiltonian(pxp)
# Hp[14].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[14].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[14].model = np.array([[0,0,1,0,4,0],[0,4,0,1,0,0]])
# Hp[14].model_coef = np.array([1,1])
# Hp[14].uc_size = np.array([3,3])
# Hp[14].uc_pos = np.array([2,2])

# Hp[15] = Hamiltonian(pxp)
# Hp[15].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[15].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[15].model = np.array([[0,0,1,0,3,0,0],[0,0,3,0,1,0,0]])
# Hp[15].model_coef = np.array([1,1])
# Hp[15].uc_size = np.array([3,3])
# Hp[15].uc_pos = np.array([2,1])

# Hp[16] = Hamiltonian(pxp)
# Hp[16].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[16].model = np.array([[0,0,0,1,0],[0,1,0,0,0]])
# Hp[16].model_coef = np.array([1,1])
# Hp[16].uc_size = np.array([3,3])
# Hp[16].uc_pos = np.array([1,1])

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
    z=zm_state(3,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(2.5,3.5))
    f = -fidelity_eval(psi_energy,e,res.x)
    print(coef,f)
    # print(f)
    if res.x <1e-5:
        return 1000
    if (np.abs(coef)>0.5).any():
        return 1000
    return -f

res = minimize(lambda coef:fidelity_error(coef),method="powell",x0=np.ones(2)*0.01)
coef = res.x
# coef = np.load("./pxp,z3,2nd_order_pert,coef,18.npy")
print(coef)
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
# Hp_total = H_operations.add(Hp_total,Hp[1],np.array([1,coef]))
Hp = Hp_total.sector.matrix()
Hm = np.conj(np.transpose(Hp))

H = Hp+Hm
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 1/2*com(Hp,Hm)

e,u = np.linalg.eigh(H)
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
z=zm_state(3,1,pxp)
psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

# res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(2.5,3.5))
# print(res.x,fidelity_eval(psi_energy,e,res.x))

for n in range(0,np.size(t,axis=0)):
    f[n] = -fidelity_eval(psi_energy,e,t[n])
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$PXP, Z_3$, Second order pertubations, N="+str(pxp.N))
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
plt.ylabel(r"$\log(\vert \langle Z_3 \vert E \rangle \vert^2)$")
plt.title(r"$PXP, Z_3$, Second order pertubations, N="+str(pxp.N))
plt.show()

#check harmonic spacing
def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

z=zm_state(3,1,pxp)

z0 = z.prod_basis()
z1= norm(np.dot(Hp,z0))
z2= norm(np.dot(Hp,z1))
z3= norm(np.dot(Hp,z2))
z4= norm(np.dot(Hp,z3))
z5= norm(np.dot(Hp,z4))
z6= norm(np.dot(Hp,z5))
z7= norm(np.dot(Hp,z6))
z8= norm(np.dot(Hp,z7))
z9= norm(np.dot(Hp,z8))
z10= norm(np.dot(Hp,z9))
z11= norm(np.dot(Hp,z10))
z12= norm(np.dot(Hp,z11))

e = []
e = np.append(e,exp(Hz,z0))
e = np.append(e,exp(Hz,z1))
e = np.append(e,exp(Hz,z2))
e = np.append(e,exp(Hz,z3))
e = np.append(e,exp(Hz,z4))
e = np.append(e,exp(Hz,z5))
e = np.append(e,exp(Hz,z6))
e = np.append(e,exp(Hz,z7))
e = np.append(e,exp(Hz,z8))
e = np.append(e,exp(Hz,z9))
e = np.append(e,exp(Hz,z10))
e = np.append(e,exp(Hz,z11))
e = np.append(e,exp(Hz,z12))

var_vals = []
var_vals = np.append(var_vals,var(Hz,z0))
var_vals = np.append(var_vals,var(Hz,z1))
var_vals = np.append(var_vals,var(Hz,z2))
var_vals = np.append(var_vals,var(Hz,z3))
var_vals = np.append(var_vals,var(Hz,z4))
var_vals = np.append(var_vals,var(Hz,z5))
var_vals = np.append(var_vals,var(Hz,z6))
var_vals = np.append(var_vals,var(Hz,z7))
var_vals = np.append(var_vals,var(Hz,z8))
var_vals = np.append(var_vals,var(Hz,z9))
var_vals = np.append(var_vals,var(Hz,z10))
var_vals = np.append(var_vals,var(Hz,z11))
var_vals = np.append(var_vals,var(Hz,z12))

e_diff = np.zeros(np.size(e)-1)
for n in range(0,np.size(e_diff,axis=0)):
    e_diff[n] = e[n+1]-e[n]
    print(e_diff[n])
    

print("\n")
print(exp(Hz,z0),var(Hz,z0))
print(exp(Hz,z1),var(Hz,z1))
print(exp(Hz,z2),var(Hz,z2))
print(exp(Hz,z3),var(Hz,z3))
print(exp(Hz,z4),var(Hz,z4))
print(exp(Hz,z5),var(Hz,z5))
print(exp(Hz,z6),var(Hz,z6))
print(exp(Hz,z7),var(Hz,z7))
print(exp(Hz,z8),var(Hz,z8))
print(exp(Hz,z9),var(Hz,z9))
print(exp(Hz,z10),var(Hz,z10))
print(exp(Hz,z11),var(Hz,z11))
print(exp(Hz,z12),var(Hz,z12))
# np.savetxt("pxp,z3,2nd_order_perts,Hz_eigvalues,"+str(pxp.N),e)
# np.savetxt("pxp,z3,2nd_order_perts,Hz_var,"+str(pxp.N),var_vals)
# np.savetxt("pxp,z3,2nd_order_perts,Hz_eig_diffs,"+str(pxp.N),e_diff)
