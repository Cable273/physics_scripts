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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=4),PT(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,2,0],[0,1,0],[0,1,0],[0,1,0]])
Hp[0].model_coef = np.array([1,1,1,1])
Hp[0].uc_size = np.array([4,4,4,4])
Hp[0].uc_pos = np.array([3,0,1,2])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[1].model_coef = np.array([1,1])
Hp[1].uc_size = np.array([4,4])
Hp[1].uc_pos = np.array([3,1])

z=zm_state(4,1,pxp)
for n in range(0,len(Hp)):
    Hp[n].gen()
Hp_total = H_operations.add(Hp[0],Hp[1], np.array([1,-1]))

H = H_operations.add(Hp_total,Hp_total.herm_conj(),np.array([1,1]))
    
Hx = (Hp_total.sector.matrix() + Hp_total.herm_conj().sector.matrix())/2
Hy = (Hp_total.sector.matrix() - Hp_total.herm_conj().sector.matrix())/(2j)
Hz = 1/2 * com (Hp_total.sector.matrix(),Hp_total.herm_conj().sector.matrix())

from Calculations import gen_fsa_basis
fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),pxp.N)
Hx_subspace = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hx,fsa_basis))
Hy_subspace = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hy,fsa_basis))
Hz_subspace = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hz,fsa_basis))

R = sp.linalg.expm(Hy*(1j)*math.pi/2)
scar_basis = np.dot(np.conj(np.transpose(R)),fsa_basis)
scar_e = np.zeros(np.size(scar_basis,axis=1))
scar_entropy = np.zeros(np.size(scar_basis,axis=1))
ent = entropy(pxp)
for n in range(0,np.size(scar_e,axis=0)):
    scar_e[n] = exp(H.sector.matrix(),scar_basis[:,n])
    scar_entropy[n] = ent.eval(scar_basis[:,n])
    
H.sector.find_eig()
ent_vals = np.zeros(pxp.dim)
for n in range(0,np.size(ent_vals,axis=0)):
    ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
plt.scatter(H.sector.eigvalues(),ent_vals)
plt.scatter(scar_e,scar_entropy,marker="x",color="red",s=100)
plt.show()

# #try find shiraishi form
# P = np.eye(pxp.dim)
# for n in range(0,np.size(fsa_basis,axis=1)):
    # P = P-np.outer(fsa_basis[:,n],fsa_basis[:,n])

