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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,gen_fsa_basis

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

N=8
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

Hp[1] = Hamiltonian(pxp)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,2,1,2,0]])
Hp[1].model_coef = np.array([1])


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
    z=zm_state(2,1,pxp,1)
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

from Calculations import get_top_band_indices,gram_schmidt
def spacing_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    Hp0 = Hp_total.sector.matrix()
    Hm0 = np.conj(np.transpose(Hp0))

    Hz = 1/2*com(Hp0,Hm0)

    #find lowest weight state
    e,u = np.linalg.eigh(Hz)
    lowest_weight = u[:,0]

    z=zm_state(2,1,pxp,1)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),pxp.N)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis

    if np.size(np.shape(fsa_basis))==2:
        Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
        for n in range(0,np.size(Hz_exp,axis=0)):
            Hz_exp[n] = exp(Hz,fsa_basis[:,n])

        Hz_diff = np.zeros(np.size(Hz_exp)-1)
        for n in range(0,np.size(Hz_diff,axis=0)):
            Hz_diff[n] = np.abs(Hz_exp[n+1] - Hz_exp[n])
            
        #spacing error
        error_matrix = np.zeros((np.size(Hz_diff),np.size(Hz_diff)))
        for n in range(0,np.size(error_matrix,axis=0)):
            for m in range(0,np.size(error_matrix,axis=0)):
                error_matrix[n,m] = np.abs(Hz_diff[n] - Hz_diff[m])
        error = np.power(np.trace(np.dot(error_matrix,np.conj(np.transpose(error_matrix)))),0.5)
        print(coef,error)
        return error
    else:
        return 1000

def var_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    Hz = 1/2 * com(Hp_total.sector.matrix(),Hm)
    z=zm_state(2,1,pxp,1)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),pxp.N)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis
    Hz_var = np.zeros(np.size(fsa_basis,axis=1))
    for n in range(0,np.size(Hz_var,axis=0)):
        Hz_var[n] = var(Hz,fsa_basis[:,n])
    error = np.max(Hz_var)
    print(coef,error)
    return error

# coef = np.ones(1)*0.1
coef = np.array([0.122959959])
# coef = np.array([0])
# # # coef = np.array([0.00152401,-1.42847732,0.09801341,0.09784458])
# res = minimize(lambda coef:fidelity_error(coef),method="powell",x0=coef)
# res = minimize(lambda coef:var_error(coef),method="Nelder-Mead",x0=coef)
# # res = minimize(lambda coef:spacing_error(coef),method="Nelder-Mead",x0=[0])
# coef = res.x

# coef = np.zeros(1)
# # coef = np.load("./data/pxp,z4,pert_coef.npy")
# # coef = res.x
# # coef[1] = -1

# error = spacing_error(coef)
# print("ERROR VALUE: "+str(error))

print("COEF")
print(coef)
print("\n")
Hp_total = deepcopy(Hp[0])
# for n in range(1,len(Hp)):
    # Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hp_total = H_operations.add(Hp_total,Hp[1],np.array([1,coef]))
Hp = Hp_total.sector.matrix()
Hm = np.conj(np.transpose(Hp))

H = Hp+Hm
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 1/2*com(Hp,Hm)
e,u = np.linalg.eigh(Hz)
z=zm_state(2,1,pxp,1)
print(z.bits)
print("|<lw Hz | z2 > |")
print(np.abs(np.vdot(u[:,0],z.prod_basis())))

e,u = np.linalg.eigh(H)

Hz_e,Hz_u = np.linalg.eigh(Hz)
z=zm_state(2,1,pxp,1)
lowest_weight = Hz_u[:,0]
print(np.vdot(lowest_weight,z.prod_basis()))

t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
z=zm_state(2,1,pxp,1)
psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(2.2,3.0))
f0 = fidelity_eval(psi_energy,e,res.x)
print("F0 REVIVAL")
print(res.x,fidelity_eval(psi_energy,e,res.x))
# np.savetxt("pxp,z4,f0,"+str(pxp.N),[f0])

for n in range(0,np.size(t,axis=0)):
    f[n] = -fidelity_eval(psi_energy,e,t[n])
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$PXP, Z_2$, PXXXP Bipartite $H^+$, level spacing pert optimization, N="+str(pxp.N))
# plt.title(r"$PXP, Z_2$, PXXXP Bipartite $H^+$, Var pert optimization, N="+str(pxp.N))
# plt.title(r"$PXP, Z_2$, PXXXP Single Site $H^+$, Var pert optimization, N="+str(pxp.N))
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
plt.title(r"$PXP, Z_2$, PXXXP Bipartite $H^+$, level spacing pert optimization, N="+str(pxp.N))
# plt.title(r"$PXP, Z_2$, PXXXP Bipartite $H^+$, Var pert optimization, N="+str(pxp.N))
# plt.title(r"$PXP, Z_2$, PXXXP Single Site $H^+$, Var pert optimization, N="+str(pxp.N))
plt.show()

#check harmonic spacing

z=zm_state(2,1,pxp,1)

from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)
gs = gram_schmidt(fsa_basis)
gs.ortho()
fsa_basis = gs.ortho_basis
    
Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
for n in range(0,np.size(fsa_basis,axis=1)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])
    print(Hz_exp[n],Hz_var[n])

print("\n")
e_diff = np.zeros(np.size(Hz_exp)-1)
for n in range(0,np.size(e_diff,axis=0)):
    e_diff[n] = Hz_exp[n+1]-Hz_exp[n]
    print(e_diff[n])

#check revivals in projected subspace
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H,fsa_basis))
plt.matshow(np.abs(H_fsa))
plt.show()
e,u = np.linalg.eigh(H_fsa)
psi_energy = np.conj(u[0,:])
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.show()

# np.save("z4,Hz_diff,16",e_diff)
    
# np.savetxt("pxp,pxxxp,z2,pert_coef,"+str(pxp.N),[coef])
# np.savetxt("pxp,pxxxp,z2,Hz_eigvalues,"+str(pxp.N),Hz_exp)
# np.savetxt("pxp,pxxxp,z2,Hz_var,"+str(pxp.N),Hz_var)
# np.savetxt("pxp,pxxxp,z2,Hz_eig_diffs,"+str(pxp.N),e_diff)
# np.savetxt("pxp,pxxxp,z2,f0,"+str(pxp.N),[f0])
# np.savetxt("pxp,pxxxp,z2,spacing_error,"+str(pxp.N),[error])
