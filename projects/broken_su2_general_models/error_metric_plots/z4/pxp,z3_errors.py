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
from Symmetry_Classes import *
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
N=16
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

#1st order
Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,1,2,1,0]])
Hp[1].model_coef = np.array([1])
Hp[1].uc_size = np.array([4])
Hp[1].uc_pos = np.array([0])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[2].model_coef = np.array([1,1])
Hp[2].uc_size = np.array([4,4])
Hp[2].uc_pos = np.array([3,1])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([4,4,4,4])
Hp[3].uc_pos = np.array([0,0,1,1])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Hp[4].model_coef = np.array([1,1,1,1])
Hp[4].uc_size = np.array([4,4,4,4])
Hp[4].uc_pos = np.array([2,3,3,2])

for n in range(0,len(Hp)):
    Hp[n].gen()

def subspace_variance(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()

    H = H_operations.add(Hp_total,Hm,np.array([1,1]))
    H.sector.find_eig()

    z=zm_state(4,1,pxp)
    from Calculations import gen_fsa_basis
    fsa_dim = int(2*pxp.N/4)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),fsa_dim)
    # for n in range(0,np.size(fsa_basis,axis=1)):
        # for m in range(0,np.size(fsa_basis,axis=1)):
            # temp = np.abs(np.vdot(fsa_basis[:,n],fsa_basis[:,m]))
            # if temp > 1e-5:
                # print(temp,n,m)
    fsa_basis,temp = np.linalg.qr(fsa_basis)

    H2 = np.dot(H.sector.matrix(),H.sector.matrix())
    H2_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H2,fsa_basis))
    H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    return subspace_variance
def max_variance(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = Hp_total.herm_conj()
    Hz = 1/2 * com(Hp_total.sector.matrix(),Hm.sector.matrix())

    z=zm_state(4,1,pxp)
    from Calculations import gen_fsa_basis
    fsa_dim = int(2*pxp.N/4)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),fsa_dim)
    fsa_basis,temp = np.linalg.qr(fsa_basis)

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

    H = H_operations.add(Hp_total,Hm,np.array([1,1]))
    H.sector.find_eig()

    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),z.prod_basis())

    t=np.arange(0,6,0.001)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(),t[n])
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
    Hz = 1/2 * com(Hp_total.sector.matrix(),Hm.sector.matrix())

    z=zm_state(4,1,pxp)
    from Calculations import gen_fsa_basis
    fsa_dim = int(2*pxp.N/4)
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),z.prod_basis(),fsa_dim)
    fsa_basis,temp = np.linalg.qr(fsa_basis)

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

coef = np.zeros(4)
errors = np.zeros((3,4))
errors[0,0]= fidelity_erorr(coef)
errors[0,1]= subspace_variance(coef)
errors[0,2]= max_variance(coef)
errors[0,3]= spacing_error(coef)

coef = np.load("../../../pxp,2nd_order_perts/z4_first_order/data/pxp,z4,pert_coef.npy")
errors[1,0]= fidelity_erorr(coef)
errors[1,1]= subspace_variance(coef)
errors[1,2]= max_variance(coef)
errors[1,3]= spacing_error(coef)


#2nd order
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
Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,2,0,0],[0,0,2,0]])
Hp[1].model_coef = np.array([1,1,1,1])
Hp[1].uc_size = np.array([4,4,4,4])
Hp[1].uc_pos = np.array([3,2,3,2])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].model = np.array([[0,0,1,0,0],[0,0,1,0,0]])
Hp[2].model_coef = np.array([1,1])
Hp[2].uc_size = np.array([4,4])
Hp[2].uc_pos = np.array([3,1])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].model = np.array([[0,0,0,1,0],[0,1,0,0,0]])
Hp[3].model_coef = np.array([1,1])
Hp[3].uc_size = np.array([4,4])
Hp[3].uc_pos = np.array([3,1])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
Hp[4].model = np.array([[0,0,4,0,1,0],[0,1,0,4,0,0]])
Hp[4].model_coef = np.array([1,1])
Hp[4].uc_size = np.array([4,4])
Hp[4].uc_pos = np.array([3,0])

Hp[5] = Hamiltonian(pxp,pxp_syms)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
Hp[5].model = np.array([[0,2,0,4,0],[0,4,0,2,0]])
Hp[5].model_coef = np.array([1,1])
Hp[5].uc_size = np.array([4,4])
Hp[5].uc_pos = np.array([3,1])

Hp[5] = Hamiltonian(pxp,pxp_syms)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
Hp[5].model = np.array([[0,2,0,3,0],[0,3,0,2,0]])
Hp[5].model_coef = np.array([1,1])
Hp[5].uc_size = np.array([4,4])
Hp[5].uc_pos = np.array([3,1])

Hp[6] = Hamiltonian(pxp,pxp_syms)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
Hp[6].model = np.array([[0,2,0,3,0,0],[0,0,3,0,2,0]])
Hp[6].model_coef = np.array([1,1])
Hp[6].uc_size = np.array([4,4])
Hp[6].uc_pos = np.array([3,0])

Hp[6] = Hamiltonian(pxp,pxp_syms)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
Hp[6].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
Hp[6].model_coef = np.array([1,1])
Hp[6].uc_size = np.array([4,4])
Hp[6].uc_pos = np.array([3,1])

Hp[7] = Hamiltonian(pxp,pxp_syms)
Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[7].site_ops[4] = np.array([[0,0],[0,1]])
Hp[7].model = np.array([[0,1,0,0,0],[0,0,0,1,0]])
Hp[7].model_coef = np.array([1,1])
Hp[7].uc_size = np.array([4,4])
Hp[7].uc_pos = np.array([2,2])

Hp[8] = Hamiltonian(pxp,pxp_syms)
Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[8].site_ops[4] = np.array([[0,0],[0,1]])
Hp[8].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0]])
Hp[8].model_coef = np.array([1,1])
Hp[8].uc_size = np.array([4,4])
Hp[8].uc_pos = np.array([1,2])

Hp[9] = Hamiltonian(pxp,pxp_syms)
Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[9].site_ops[4] = np.array([[0,0],[0,1]])
Hp[9].model = np.array([[0,3,0,1,0,0],[0,0,1,0,3,0]])
Hp[9].model_coef = np.array([1,1])
Hp[9].uc_size = np.array([4,4])
Hp[9].uc_pos = np.array([3,0])

Hp[10] = Hamiltonian(pxp,pxp_syms)
Hp[10].site_ops[1] = np.array([[0,0],[1,0]])
Hp[10].site_ops[2] = np.array([[0,1],[0,0]])
Hp[10].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[10].site_ops[4] = np.array([[0,0],[0,1]])
Hp[10].model = np.array([[0,0,2,0,0]])
Hp[10].model_coef = np.array([1])
Hp[10].uc_size = np.array([4])
Hp[10].uc_pos = np.array([2])

Hp[11] = Hamiltonian(pxp,pxp_syms)
Hp[11].site_ops[1] = np.array([[0,0],[1,0]])
Hp[11].site_ops[2] = np.array([[0,1],[0,0]])
Hp[11].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[11].site_ops[4] = np.array([[0,0],[0,1]])
Hp[11].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
Hp[11].model_coef = np.array([1,1,1,1])
Hp[11].uc_size = np.array([4,4,4,4])
Hp[11].uc_pos = np.array([0,0,1,1])

Hp[12] = Hamiltonian(pxp,pxp_syms)
Hp[12].site_ops[1] = np.array([[0,0],[1,0]])
Hp[12].site_ops[2] = np.array([[0,1],[0,0]])
Hp[12].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[12].site_ops[4] = np.array([[0,0],[0,1]])
Hp[12].model = np.array([[0,0,1,0,0]])
Hp[12].model_coef = np.array([1])
Hp[12].uc_size = np.array([4])
Hp[12].uc_pos = np.array([0])

Hp[13] = Hamiltonian(pxp,pxp_syms)
Hp[13].site_ops[1] = np.array([[0,0],[1,0]])
Hp[13].site_ops[2] = np.array([[0,1],[0,0]])
Hp[13].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[13].site_ops[4] = np.array([[0,0],[0,1]])
Hp[13].model = np.array([[0,1,0,0,0],[0,0,0,1,0]])
Hp[13].model_coef = np.array([1,1])
Hp[13].uc_size = np.array([4,4])
Hp[13].uc_pos = np.array([0,0])

Hp[14] = Hamiltonian(pxp,pxp_syms)
Hp[14].site_ops[1] = np.array([[0,0],[1,0]])
Hp[14].site_ops[2] = np.array([[0,1],[0,0]])
Hp[14].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[14].site_ops[4] = np.array([[0,0],[0,1]])
Hp[14].model = np.array([[0,1,0,4,0],[0,4,0,1,0]])
Hp[14].model_coef = np.array([1,1])
Hp[14].uc_size = np.array([4,4])
Hp[14].uc_pos = np.array([0,0])

Hp[15] = Hamiltonian(pxp,pxp_syms)
Hp[15].site_ops[1] = np.array([[0,0],[1,0]])
Hp[15].site_ops[2] = np.array([[0,1],[0,0]])
Hp[15].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[15].site_ops[4] = np.array([[0,0],[0,1]])
Hp[15].model = np.array([[0,0,4,0,2,0],[0,2,0,4,0,0]])
Hp[15].model_coef = np.array([1,1])
Hp[15].uc_size = np.array([4,4])
Hp[15].uc_pos = np.array([0,3])

Hp[16] = Hamiltonian(pxp,pxp_syms)
Hp[16].site_ops[1] = np.array([[0,0],[1,0]])
Hp[16].site_ops[2] = np.array([[0,1],[0,0]])
Hp[16].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[16].site_ops[4] = np.array([[0,0],[0,1]])
Hp[16].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0]])
Hp[16].model_coef = np.array([1,1])
Hp[16].uc_size = np.array([4,4])
Hp[16].uc_pos = np.array([0,3])

Hp[17] = Hamiltonian(pxp,pxp_syms)
Hp[17].site_ops[1] = np.array([[0,0],[1,0]])
Hp[17].site_ops[2] = np.array([[0,1],[0,0]])
Hp[17].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[17].site_ops[4] = np.array([[0,0],[0,1]])
Hp[17].model = np.array([[0,4,0,1,0],[0,1,0,4,0]])
Hp[17].model_coef = np.array([1,1])
Hp[17].uc_size = np.array([4,4])
Hp[17].uc_pos = np.array([3,1])

Hp[18] = Hamiltonian(pxp,pxp_syms)
Hp[18].site_ops[1] = np.array([[0,0],[1,0]])
Hp[18].site_ops[2] = np.array([[0,1],[0,0]])
Hp[18].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[18].site_ops[4] = np.array([[0,0],[0,1]])
Hp[18].model = np.array([[0,0,3,0,1,0],[0,1,0,3,0,0]])
Hp[18].model_coef = np.array([1,1])
Hp[18].uc_size = np.array([4,4])
Hp[18].uc_pos = np.array([2,1])

Hp[19] = Hamiltonian(pxp,pxp_syms)
Hp[19].site_ops[1] = np.array([[0,0],[1,0]])
Hp[19].site_ops[2] = np.array([[0,1],[0,0]])
Hp[19].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[19].site_ops[4] = np.array([[0,0],[0,1]])
Hp[19].model = np.array([[0,2,0,0,0],[0,0,0,2,0]])
Hp[19].model_coef = np.array([1,1])
Hp[19].uc_size = np.array([4,4])
Hp[19].uc_pos = np.array([3,1])

Hp[20] = Hamiltonian(pxp,pxp_syms)
Hp[20].site_ops[1] = np.array([[0,0],[1,0]])
Hp[20].site_ops[2] = np.array([[0,1],[0,0]])
Hp[20].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[20].site_ops[4] = np.array([[0,0],[0,1]])
Hp[20].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
Hp[20].model_coef = np.array([1,1])
Hp[20].uc_size = np.array([4,4])
Hp[20].uc_pos = np.array([0,0])

Hp[21] = Hamiltonian(pxp,pxp_syms)
Hp[21].site_ops[1] = np.array([[0,0],[1,0]])
Hp[21].site_ops[2] = np.array([[0,1],[0,0]])
Hp[21].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[21].site_ops[4] = np.array([[0,0],[0,1]])
Hp[21].model = np.array([[0,3,0,1,0,0],[0,0,1,0,3,0]])
Hp[21].model_coef = np.array([1,1])
Hp[21].uc_size = np.array([4,4])
Hp[21].uc_pos = np.array([0,3])

Hp[22] = Hamiltonian(pxp,pxp_syms)
Hp[22].site_ops[1] = np.array([[0,0],[1,0]])
Hp[22].site_ops[2] = np.array([[0,1],[0,0]])
Hp[22].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[22].site_ops[4] = np.array([[0,0],[0,1]])
Hp[22].model = np.array([[0,0,2,0,0,0],[0,0,0,2,0,0]])
Hp[22].model_coef = np.array([1,1])
Hp[22].uc_size = np.array([4,4])
Hp[22].uc_pos = np.array([2,1])

Hp[23] = Hamiltonian(pxp,pxp_syms)
Hp[23].site_ops[1] = np.array([[0,0],[1,0]])
Hp[23].site_ops[2] = np.array([[0,1],[0,0]])
Hp[23].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[23].site_ops[4] = np.array([[0,0],[0,1]])
Hp[23].model = np.array([[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,3,0,1,0],[0,1,0,3,0,0]])
Hp[23].model_coef = np.array([1,1,1,1])
Hp[23].uc_size = np.array([4,4,4,4])
Hp[23].uc_pos = np.array([0,1,3,2])

Hp[24] = Hamiltonian(pxp,pxp_syms)
Hp[24].site_ops[1] = np.array([[0,0],[1,0]])
Hp[24].site_ops[2] = np.array([[0,1],[0,0]])
Hp[24].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[24].site_ops[4] = np.array([[0,0],[0,1]])
Hp[24].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
Hp[24].model_coef = np.array([1,1])
Hp[24].uc_size = np.array([4,4])
Hp[24].uc_pos = np.array([2,2])

Hp[25] = Hamiltonian(pxp,pxp_syms)
Hp[25].site_ops[1] = np.array([[0,0],[1,0]])
Hp[25].site_ops[2] = np.array([[0,1],[0,0]])
Hp[25].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[25].site_ops[4] = np.array([[0,0],[0,1]])
Hp[25].model = np.array([[0,0,1,0,3,0],[0,3,0,1,0,0]])
Hp[25].model_coef = np.array([1,1])
Hp[25].uc_size = np.array([4,4])
Hp[25].uc_pos = np.array([1,2])

Hp[26] = Hamiltonian(pxp,pxp_syms)
Hp[26].site_ops[1] = np.array([[0,0],[1,0]])
Hp[26].site_ops[2] = np.array([[0,1],[0,0]])
Hp[26].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[26].site_ops[4] = np.array([[0,0],[0,1]])
Hp[26].model = np.array([[0,0,0,1,0,0],[0,0,1,0,0,0]])
Hp[26].model_coef = np.array([1,1])
Hp[26].uc_size = np.array([4,4])
Hp[26].uc_pos = np.array([0,3])

Hp[27] = Hamiltonian(pxp,pxp_syms)
Hp[27].site_ops[1] = np.array([[0,0],[1,0]])
Hp[27].site_ops[2] = np.array([[0,1],[0,0]])
Hp[27].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[27].site_ops[4] = np.array([[0,0],[0,1]])
Hp[27].model = np.array([[0,0,2,0,3,0],[0,3,0,2,0,0]])
Hp[27].model_coef = np.array([1,1])
Hp[27].uc_size = np.array([4,4])
Hp[27].uc_pos = np.array([2,1])

Hp[28] = Hamiltonian(pxp,pxp_syms)
Hp[28].site_ops[1] = np.array([[0,0],[1,0]])
Hp[28].site_ops[2] = np.array([[0,1],[0,0]])
Hp[28].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[28].site_ops[4] = np.array([[0,0],[0,1]])
Hp[28].model = np.array([[0,0,4,0,2,0,0],[0,0,2,0,4,0,0],[0,1,0,4,0,4,0],[0,4,0,4,0,1,0]])
Hp[28].model_coef = np.array([1,1,1,1])
Hp[28].uc_size = np.array([4,4,4,4])
Hp[28].uc_pos = np.array([0,2,2,0])

Hp[29] = Hamiltonian(pxp,pxp_syms)
Hp[29].site_ops[1] = np.array([[0,0],[1,0]])
Hp[29].site_ops[2] = np.array([[0,1],[0,0]])
Hp[29].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[29].site_ops[4] = np.array([[0,0],[0,1]])
Hp[29].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
Hp[29].model_coef = np.array([1,1])
Hp[29].uc_size = np.array([4,4])
Hp[29].uc_pos = np.array([0,3])

Hp[30] = Hamiltonian(pxp,pxp_syms)
Hp[30].site_ops[1] = np.array([[0,0],[1,0]])
Hp[30].site_ops[2] = np.array([[0,1],[0,0]])
Hp[30].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[30].site_ops[4] = np.array([[0,0],[0,1]])
Hp[30].model = np.array([[0,0,3,0,2,0,0],[0,0,2,0,3,0,0]])
Hp[30].model_coef = np.array([1,1])
Hp[30].uc_size = np.array([4,4])
Hp[30].uc_pos = np.array([0,2])

Hp[31] = Hamiltonian(pxp,pxp_syms)
Hp[31].site_ops[1] = np.array([[0,0],[1,0]])
Hp[31].site_ops[2] = np.array([[0,1],[0,0]])
Hp[31].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[31].site_ops[4] = np.array([[0,0],[0,1]])
Hp[31].model = np.array([[0,0,3,0,1,0,0],[0,0,1,0,3,0,0]])
Hp[31].model_coef = np.array([1,1])
Hp[31].uc_size = np.array([4,4])
Hp[31].uc_pos = np.array([1,1])

Hp[32] = Hamiltonian(pxp,pxp_syms)
Hp[32].site_ops[1] = np.array([[0,0],[1,0]])
Hp[32].site_ops[2] = np.array([[0,1],[0,0]])
Hp[32].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[32].site_ops[4] = np.array([[0,0],[0,1]])
Hp[32].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
Hp[32].model_coef = np.array([1,1])
Hp[32].uc_size = np.array([4,4])
Hp[32].uc_pos = np.array([2,1])

Hp[33] = Hamiltonian(pxp,pxp_syms)
Hp[33].site_ops[1] = np.array([[0,0],[1,0]])
Hp[33].site_ops[2] = np.array([[0,1],[0,0]])
Hp[33].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[33].site_ops[4] = np.array([[0,0],[0,1]])
Hp[33].model = np.array([[0,0,3,0,1,0,0],[0,0,1,0,3,0,0]])
Hp[33].model_coef = np.array([1,1])
Hp[33].uc_size = np.array([4,4])
Hp[33].uc_pos = np.array([3,3])

Hp[34] = Hamiltonian(pxp,pxp_syms)
Hp[34].site_ops[1] = np.array([[0,0],[1,0]])
Hp[34].site_ops[2] = np.array([[0,1],[0,0]])
Hp[34].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[34].site_ops[4] = np.array([[0,0],[0,1]])
Hp[34].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
Hp[34].model_coef = np.array([1,1])
Hp[34].uc_size = np.array([4,4])
Hp[34].uc_pos = np.array([1,2])

Hp[35] = Hamiltonian(pxp,pxp_syms)
Hp[35].site_ops[1] = np.array([[0,0],[1,0]])
Hp[35].site_ops[2] = np.array([[0,1],[0,0]])
Hp[35].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[35].site_ops[4] = np.array([[0,0],[0,1]])
Hp[35].model = np.array([[0,0,1,0,3,0,0],[0,0,3,0,1,0,0]])
Hp[35].model_coef = np.array([1,1])
Hp[35].uc_size = np.array([4,4])
Hp[35].uc_pos = np.array([0,2])

Hp[36] = Hamiltonian(pxp,pxp_syms)
Hp[36].site_ops[1] = np.array([[0,0],[1,0]])
Hp[36].site_ops[2] = np.array([[0,1],[0,0]])
Hp[36].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[36].site_ops[4] = np.array([[0,0],[0,1]])
Hp[36].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[36].model_coef = np.array([1,1])
Hp[36].uc_size = np.array([4,4])
Hp[36].uc_pos = np.array([3,1])


for n in range(0,len(Hp)):
    Hp[n].gen()

coef = np.load("../../../pxp,2nd_order_perts/z4_2nd_order/pxp,z4,2nd_order_pert_coef,12.npy")
print(np.size(coef))
print(coef)
errors[2,0]= fidelity_erorr(coef)
errors[2,1]= subspace_variance(coef)
errors[2,2]= max_variance(coef)
errors[2,3]= spacing_error(coef)

print(errors)
np.save("pxp,z3,su2_errors,"+str(pxp.N),errors)
