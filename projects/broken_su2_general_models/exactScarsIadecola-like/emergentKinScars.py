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
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=6
pxp = unlocking_System([0,1],"periodic",2,N,)
pxp.gen_basis()

S = 1/2
m = np.arange(-S,S)
couplings = np.power(S*(S+1)-m*(m+1),0.5)
Jp = np.diag(couplings,1)
Jm = np.diag(couplings,-1)
x = 1/2 *(Jp + Jm)
y = 1/2j *(Jp - Jm)
z = 1/2 *com(Jp,Jm)
z2 = np.dot(z,z)

#create Hamiltonian
J = 1
h = 0.1
D = 0.1
H = Hamiltonian(pxp)
H.site_ops[1] = x
H.site_ops[2] = y
H.site_ops[3] = z
H.site_ops[4] = z2
H.model = np.array([[1,1],[2,2],[3],[4]])
H.model_coef = np.array([J,J,h,D])
H.gen()

Hp = np.zeros((pxp.dim,pxp.dim))
for n in range(0,np.size(pxp.basis,axis=0)):
    bits = np.copy(pxp.basis[n])
    for m in range(0,pxp.N):
        if bits[m] == 2:
            new_bits = np.copy(bits)
            new_bits[m] = 0
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            Hp[pxp.keys[new_ref],n] += 2 * (-1)**m
Hm = np.conj(np.transpose(Hp))
Hx = 1/2 *(Hp+Hm)

#scarred eigenstates
psi = ref_state(0,pxp).prod_basis()
from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hm,psi,pxp.N)

H0 = Hamiltonian(pxp)
H0.site_ops[1] = x
H0.site_ops[2] = y
H0.site_ops[3] = z
H0.site_ops[4] = z2
H0.model = np.array([[1,1],[2,2]])
H0.model_coef = np.array([J,J])
H0.gen()

H1 = Hamiltonian(pxp)
H1.site_ops[1] = x
H1.site_ops[2] = y
H1.site_ops[3] = z
H1.site_ops[4] = z2
H1.model = np.array([[3],[4]])
H1.model_coef = np.array([h,D])
H1.gen()

print("<H0> scar states")
for n in range(0,np.size(fsa_basis,axis=1)):
    print(exp(H0.sector.matrix(),fsa_basis[:,n]),var(H0.sector.matrix(),fsa_basis[:,n]))

print("\n")
print("<H1> scar states")
for n in range(0,np.size(fsa_basis,axis=1)):
    print(exp(H1.sector.matrix(),fsa_basis[:,n]),var(H1.sector.matrix(),fsa_basis[:,n]))
    

temp = com(H1.sector.matrix(),Hp)
temp2 = com(H1.sector.matrix(),H0.sector.matrix())
temp3 = com(H0.sector.matrix(),Hp)
print("[H0,H1]=0?")
print((np.abs(temp2)<1e-5).all())
print("[H1,Q+]=aQ+?")
print((np.abs(temp*np.max(Hp)/np.max(temp)-Hp)<1e-5).all())
print("[H0,Q+]=0?")
print((np.abs(temp3)<1e-5).all())


# e,u = np.linalg.eigh(Hx)
# lw = u[:,0]
# H.sector.find_eig()
# lw_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),lw)
# t=np.arange(0,80,0.01)
# f=np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(lw_energy,H.sector.eigvalues(),t[n])
    # f[n] = np.abs(np.vdot(evolved_state,lw_energy))**2
# plt.plot(t,f)
# plt.show()
