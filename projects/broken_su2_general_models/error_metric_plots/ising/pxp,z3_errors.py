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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general
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
N=14
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2)])

J = 1
hx = 1
hz = 1
Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[0].model = np.array([[1,1],[2,2],[1,2],[2,1],[1],[2],[3]])
Hp[0].model_coef = np.array([J/4,J/4,J/4,J/4,hx/2,hx/2,hz/2])
Hp[0].uc_size = np.array([2,2,2,2,2,2,1])
Hp[0].uc_pos = np.array([0,1,0,1,0,1,0])

#1st order
# Hp[1] = Hamiltonian(pxp,pxp_syms)
# Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[1].model = np.array([[2,2,4],[2,2,0],[1,1,0],[1,1,4]])
# Hp[1].model_coef = np.array([1,1,1,1])
# Hp[1].uc_size = np.array([2,2,2,2])
# Hp[1].uc_pos = np.array([1,1,0,0])

# Hp[2] = Hamiltonian(pxp,pxp_syms)
# Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[2].model = np.array([[4,2,2],[4,2,1],[0,2,2],[0,2,1],[4,1,1],[0,1,1],[0,1,2],[4,1,2]])
# Hp[2].model_coef = np.array([1,1,1,1,1,1,1,1,1])
# Hp[2].uc_size = np.array([2,2,2,2,2,2,2,2])
# Hp[2].uc_pos = np.array([0,0,0,0,1,1,1,1])

# Hp[3] = Hamiltonian(pxp,pxp_syms)
# Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[3].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[3].model = np.array([[2,1,4],[2,1,0],[1,2,0],[1,2,4]])
# Hp[3].model_coef = np.array([1,1,1,1])
# Hp[3].uc_size = np.array([2,2,2,2])
# Hp[3].uc_pos = np.array([1,1,0,0])

# Hp[4] = Hamiltonian(pxp,pxp_syms)
# Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[4].model = np.array([[1,3,3,2],[1,3,3,1],[1,3,3,1],[2,3,3,1],[2,3,3,2],[2,3,3,2],[2,3,3,1],[1,3,3,2]])
# Hp[4].model_coef = np.array([1,1,1,1,1,1,1,1])
# Hp[4].uc_size = np.array([2,2,2,2,2,2,2,2])
# Hp[4].uc_pos = np.array([1,0,1,0,0,1,1,0])

# Hp[5] = Hamiltonian(pxp,pxp_syms)
# Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[5].model = np.array([[1,4],[1,0],[2,0],[2,4]])
# Hp[5].model_coef = np.array([1,1,1,1])
# Hp[5].uc_size = np.array([2,2,2,2])
# Hp[5].uc_pos = np.array([0,0,1,1])

# Hp[6] = Hamiltonian(pxp,pxp_syms)
# Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[6].model = np.array([[4,2],[0,2],[0,1],[4,1]])
# Hp[6].model_coef = np.array([1,1,1,1])
# Hp[6].uc_size = np.array([2,2,2,2])
# Hp[6].uc_pos = np.array([0,0,1,1])

# Hp[7] = Hamiltonian(pxp,pxp_syms)
# Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[7].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[7].model = np.array([[4,2,1],[0,2,1],[0,1,2],[4,1,2],[4,2,2],[0,2,2],[0,1,1],[4,1,1]])
# Hp[7].model_coef = np.array([1,1,1,1,1,1,1,1])
# Hp[7].uc_size = np.array([2,2,2,2,2,2,2,2])
# Hp[7].uc_pos = np.array([1,1,0,0,1,1,0,0])

# Hp[8] = Hamiltonian(pxp,pxp_syms)
# Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[8].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[8].model = np.array([[1,1,4],[1,1,0],[2,2,0],[2,2,4]])
# Hp[8].model_coef = np.array([1,1,1,1])
# Hp[8].uc_size = np.array([2,2,2,2])
# Hp[8].uc_pos = np.array([1,1,0,0])

# Hp[9] = Hamiltonian(pxp,pxp_syms)
# Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[9].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[9].model = np.array([[1,1,1],[2,2,2],[1,1,2],[2,2,1]])
# Hp[9].model_coef = np.array([1,1,1,1])
# Hp[9].uc_size = np.array([2,2,2,2])
# Hp[9].uc_pos = np.array([1,0,1,0])

# Hp[10] = Hamiltonian(pxp,pxp_syms)
# Hp[10].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[10].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[10].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[10].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[10].model = np.array([[1,3,3],[2,3,3]])
# Hp[10].model_coef = np.array([1,1])

# Hp[11] = Hamiltonian(pxp,pxp_syms)
# Hp[11].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[11].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[11].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[11].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[11].model = np.array([[3,3,1],[3,3,2]])
# Hp[11].model_coef = np.array([1,1])

# Hp[12] = Hamiltonian(pxp,pxp_syms)
# Hp[12].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[12].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[12].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[12].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[12].model = np.array([[1,3,1],[2,3,2]])
# Hp[12].model_coef = np.array([1,1])
# Hp[12].uc_size = np.array([2,2])
# Hp[12].uc_pos = np.array([1,0])

# Hp[13] = Hamiltonian(pxp,pxp_syms)
# Hp[13].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[13].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[13].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[13].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[13].model = np.array([[2,2,1],[1,1,2],[2,2,2],[1,1,1]])
# Hp[13].model_coef = np.array([1,1,1,1])
# Hp[13].uc_size = np.array([2,2,2,2])
# Hp[13].uc_pos = np.array([1,0,1,0])

# Hp[14] = Hamiltonian(pxp,pxp_syms)
# Hp[14].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[14].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[14].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[14].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[14].model = np.array([[2,1,1],[1,2,2],[1,2,1],[2,1,2]])
# Hp[14].model_coef = np.array([1,1,1,1])
# Hp[14].uc_size = np.array([2,2,2,2])
# Hp[14].uc_pos = np.array([1,0,0,1])

# Hp[15] = Hamiltonian(pxp,pxp_syms)
# Hp[15].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[15].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[15].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[15].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[15].model = np.array([[3,1],[1,3],[3,2],[2,3]])
# Hp[15].model_coef = np.array([1,1,1,1])
# Hp[15].uc_size = np.array([2,2,2,2])
# Hp[15].uc_pos = np.array([0,0,1,1])

# Hp[16] = Hamiltonian(pxp,pxp_syms)
# Hp[16].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[16].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[16].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[16].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[16].model = np.array([[2,3,1],[1,3,2]])
# Hp[16].model_coef = np.array([1,1])
# Hp[16].uc_size = np.array([2,2])
# Hp[16].uc_pos = np.array([1,0])

#no repeat
Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
Hp[1].model = np.array([[1,3,3,2],[1,3,3,1],[1,3,3,1],[2,3,3,1],[2,3,3,2],[2,3,3,2],[2,3,3,1],[1,3,3,2]])
Hp[1].model_coef = np.array([1,1,1,1,1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2,2,2,2,2])
Hp[1].uc_pos = np.array([1,0,1,0,0,1,1,0])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[1,1,1],[2,2,2],[1,1,2],[2,2,1]])
Hp[2].model_coef = np.array([1,1,1,1])
Hp[2].uc_size = np.array([2,2,2,2])
Hp[2].uc_pos = np.array([1,0,1,0])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].site_ops[4] = np.array([[0,0],[0,1]])
Hp[3].model = np.array([[1,3,3],[2,3,3]])
Hp[3].model_coef = np.array([1,1])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
Hp[4].model = np.array([[3,3,1],[3,3,2]])
Hp[4].model_coef = np.array([1,1])

Hp[5] = Hamiltonian(pxp,pxp_syms)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
Hp[5].model = np.array([[1,3,1],[2,3,2]])
Hp[5].model_coef = np.array([1,1])
Hp[5].uc_size = np.array([2,2])
Hp[5].uc_pos = np.array([1,0])

Hp[6] = Hamiltonian(pxp,pxp_syms)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
Hp[6].model = np.array([[2,2,1],[1,1,2],[2,2,2],[1,1,1]])
Hp[6].model_coef = np.array([1,1,1,1])
Hp[6].uc_size = np.array([2,2,2,2])
Hp[6].uc_pos = np.array([1,0,1,0])

Hp[7] = Hamiltonian(pxp,pxp_syms)
Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[7].site_ops[4] = np.array([[0,0],[0,1]])
Hp[7].model = np.array([[2,1,1],[1,2,2],[1,2,1],[2,1,2]])
Hp[7].model_coef = np.array([1,1,1,1])
Hp[7].uc_size = np.array([2,2,2,2])
Hp[7].uc_pos = np.array([1,0,0,1])

Hp[8] = Hamiltonian(pxp,pxp_syms)
Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[8].site_ops[4] = np.array([[0,0],[0,1]])
Hp[8].model = np.array([[3,1],[1,3],[3,2],[2,3]])
Hp[8].model_coef = np.array([1,1,1,1])
Hp[8].uc_size = np.array([2,2,2,2])
Hp[8].uc_pos = np.array([0,0,1,1])

Hp[9] = Hamiltonian(pxp,pxp_syms)
Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[9].site_ops[4] = np.array([[0,0],[0,1]])
Hp[9].model = np.array([[2,3,1],[1,3,2]])
Hp[9].model_coef = np.array([1,1])
Hp[9].uc_size = np.array([2,2])
Hp[9].uc_pos = np.array([1,0])


k=[0]
for n in range(0,len(Hp)):
    Hp[n].gen(k)

def subspace_variance(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()

    H = H_operations.add(Hp_total,Hm,np.array([1,1]))
    H.sector.find_eig(k)

    Hz = 1/2 * com(Hp_total.sector.matrix(k),Hm.sector.matrix(k))
    e,u = np.linalg.eigh(Hz)
    psi = u[:,0]

    from Calculations import gen_fsa_basis,gram_schmidt
    fsa_dim = int(pxp.N)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(k),psi,fsa_dim)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis

    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H2,fsa_basis))
    H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(k),fsa_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    return subspace_variance
def max_variance(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()
    Hz = 1/2 * com(Hp_total.sector.matrix(k),Hm.sector.matrix(k))
    e,u = np.linalg.eigh(Hz)
    psi = u[:,0]

    from Calculations import gen_fsa_basis,gram_schmidt
    fsa_dim = int(pxp.N)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(k),psi,fsa_dim)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis

    var_vals = np.zeros(np.size(fsa_basis,axis=1))
    for n in range(0,np.size(var_vals,axis=0)):
        var_vals[n] = np.real(var(Hz,fsa_basis[:,n]))
    error = np.max(var_vals)
    return error


def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return f

from scipy.optimize import minimize_scalar
def fidelity_erorr(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()
    Hz = 1/2 * com(Hp_total.sector.matrix(k),Hm.sector.matrix(k))
    e,u = np.linalg.eigh(Hz)
    psi = u[:,0]

    H = H_operations.add(Hp_total,Hm,np.array([1,1]))
    H.sector.find_eig(k)

    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),psi)

    t=np.arange(0,6,0.001)
    # t=np.arange(0,20,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(k),t[n])
        f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f0 = np.max(f[cut:])
    # plt.plot(t,f)
    # plt.show()

    # res = minimize_scalar(lambda t: -fidelity_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(2.5,5.5))
    # f0 = fidelity_eval(psi_energy,H.sector.eigvalues(),res.x)
    return 1-f0

def spacing_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()
    Hz = 1/2 * com(Hp_total.sector.matrix(k),Hm.sector.matrix(k))
    e,u = np.linalg.eigh(Hz)
    psi = u[:,0]

    from Calculations import gen_fsa_basis,gram_schmidt
    fsa_dim = int(pxp.N)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(k),psi,fsa_dim)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis

    exp_vals = np.zeros(np.size(fsa_basis,axis=1))
    for n in range(0,np.size(exp_vals,axis=0)):
        exp_vals[n] = np.real(exp(Hz,fsa_basis[:,n]))
    exp_diff = np.zeros(np.size(exp_vals)-1)
    for n in range(0,np.size(exp_diff,axis=0)):
        exp_diff[n] = exp_vals[n+1]-exp_vals[n]

    M = np.zeros((np.size(exp_diff),np.size(exp_diff)))
    for n in range(0,np.size(M,axis=0)):
        for m in range(0,np.size(M,axis=1)):
            M[n,m] = np.abs(exp_diff[n]-exp_diff[m])
    error = np.power(np.trace(np.dot(M,np.conj(np.transpose(M)))),0.5)
    return error

coef = np.zeros(16)
errors = np.zeros((2,4))
errors[0,0]= fidelity_erorr(coef)
errors[0,1]= subspace_variance(coef)
errors[0,2]= max_variance(coef)
errors[0,3]= spacing_error(coef)

coef = np.load("../../Ising/ising,pert_coef,8.npy")
errors[1,0]= fidelity_erorr(coef)
errors[1,1]= subspace_variance(coef)
errors[1,2]= max_variance(coef)
errors[1,3]= spacing_error(coef)
print(errors)
# temp = np.load("./pxyp,su2_errors,18.npy")
# print(temp)
np.save("ising,su2_errors,"+str(N),errors)

