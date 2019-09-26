#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sparse_linalg
import sys
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

N=10
pxp = unlocking_System([0,1],"periodic",2,N)
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
pxp_half = unlocking_System([0],"periodic",2,N)

# #pauli ops
X = Hamiltonian(pxp,pxp_syms)
Y = Hamiltonian(pxp,pxp_syms)
Z = Hamiltonian(pxp,pxp_syms)

X.site_ops[1] = np.array([[0,1],[1,0]])
Y.site_ops[1] = np.array([[0,-1j],[1j,0]])
Z.site_ops[1] = np.array([[-1,0],[0,1]])

X.model,X.model_coef = np.array([[1]]),np.array((1))
Y.model,Y.model_coef = np.array([[1]]),np.array((1))
Z.model,Z.model_coef = np.array([[1]]),np.array((1))

X.gen()
Y.gen()
Z.gen()

#n_i n_i+1
H0 = Hamiltonian(pxp,pxp_syms)
H0.site_ops[1] = np.array([[0,0],[0,1]])
H0.model = np.array([[1,1]])
H0.model_coef = np.array((1))

#X_n
H_kick = Hamiltonian(pxp,pxp_syms)
H_kick.site_ops[1] = np.array([[0,1],[1,0]])
H_kick.model = np.array([[1]])
H_kick.model_coef = np.array((1))

H0.gen()
H_kick.gen()
H0.sector.find_eig()
H_kick.sector.find_eig()

# tau = 2*math.pi/(4/3)
tau_vals=np.arange(0.1,5,0.1)
sum_cost=np.zeros(np.size(tau_vals))
tau_index=0
for tau in tau_vals:
    print(tau_index,np.size(tau_vals))
# tau = 0.1
    V = 0.01

    U0 = np.dot(H0.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*tau*H0.sector.eigvalues())),np.conj(np.transpose(H0.sector.eigvectors()))))
    U_kick = np.dot(H_kick.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*V*H_kick.sector.eigvalues())),np.conj(np.transpose(H_kick.sector.eigvectors()))))

    F = np.dot(U_kick,U0)
    e,u = np.linalg.eig(F)

    no_steps = 2000
    t = np.arange(0,no_steps*tau,tau)
    z=zm_state(2,1,pxp)
    original_state = z.prod_basis()
    current_state = z.prod_basis()
    f_neel_floquet = np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        f_neel_floquet[n] = np.abs(np.vdot(current_state,original_state))**2
        current_state = np.dot(F,current_state)
        current_state = current_state/np.power(np.vdot(current_state,current_state),0.5)

    #find revival freq of floquet
    # cut_index=5
    for n in range(0,np.size(f_neel_floquet,axis=0)):
        if f_neel_floquet[n]<0.1:
            cut_index = n
            break
    max_index = np.argmax(f_neel_floquet[cut_index:])
    t0_floquet = t[max_index]
    print(t0_floquet)
    # plt.plot(t,f_neel_floquet)
    # plt.show()
        

    #find scaling of pxp to best match revival freq of Neel of Floquet
    alpha =0.5
    def cost(alpha):
        H=Hamiltonian(pxp_half,"x")
        H.site_ops[1] = np.array([[0,alpha[0]],[alpha[0],0]])
        H.model = np.array([[1]])
        H.model_coef=np.array((1))
        H.gen()
        H.sector.find_eig()
        z_rydberg = zm_state(2,1,pxp_half)
        f_neel_ham = fidelity(z_rydberg,H).eval(t,z_rydberg)
        # plt.plot(t,f_neel_ham)
        # plt.show()

        cut_index = 5
        for n in range(0,np.size(f_neel_ham,axis=0)):
            if f_neel_ham[n]<0.1:
                cut_index = n
                break
        max_index = np.argmax(f_neel_ham[cut_index:])
        t0_ham = t[max_index]

        print(cost,t0_ham)
        return np.abs(t0_ham-t0_floquet)

    from scipy.optimize import minimize_scalar,minimize
    # res = minimize_scalar(lambda alpha: cost(alpha),method="golden",bracket=(0.001,0.005))
    res = minimize(lambda alpha: cost(alpha),method="powell",x0=V/tau)
    print(res)

    alpha=res.x
    H=Hamiltonian(pxp,"x")
    H.site_ops[1] = np.array([[0,alpha],[alpha,0]])
    H.model = np.array([[0,1,0]])
    H.model_coef=np.array((1))
    H.gen()
    H.sector.find_eig()
    z_rydberg = zm_state(2,1,pxp)
    f_neel_ham = fidelity(z_rydberg,H).eval(t,z_rydberg)
    # plt.plot(t,f_neel_ham)
    # plt.show()

    H_diag = np.diag(np.exp(-1j*tau*H.sector.eigvalues()))
    exp_H = np.dot(H.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*tau*H.sector.eigvalues())),np.conj(np.transpose(H.sector.eigvectors()))))

    #evaluate direct overlap of all states
    states=dict()
    for n in range(0,np.size(pxp_half.basis_refs,axis=0)):
        states[n] = ref_state(pxp_half.basis_refs[n],pxp)


    pbar=ProgressBar()
    f_ham=dict()
    f_floquet=dict()
    evolution_overlap=dict()
    for state_index in pbar(range(0,len(states))):
        evolution_overlap[state_index] = np.zeros(np.size(t))
        # print(states[state_index].bits)
        original_state = states[state_index].prod_basis()
        current_state_ham = original_state
        current_state_floquet = original_state
        f_ham[state_index] = np.zeros(np.size(t))
        f_floquet[state_index] = np.zeros(np.size(t))

        for n in range(0,np.size(t,axis=0)):
            evolution_overlap[state_index][n]=np.abs(np.vdot(current_state_ham,current_state_floquet))**2
            current_state_ham = np.dot(exp_H,current_state_ham)
            current_state_floquet = np.dot(F,current_state_floquet)
            f_ham[state_index][n] = np.abs(np.vdot(current_state_ham,original_state))**2
            f_floquet[state_index][n] = np.abs(np.vdot(current_state_floquet,original_state))**2



    #plot
    # for n in range(0,len(f_ham)):
        # plt.plot(t,f_ham[n])
    # plt.show()

    # for n in range(0,len(f_floquet)):
        # plt.plot(t,f_floquet[n])
    # plt.show()

    temp2=0
    for n in range(0,len(evolution_overlap)):
        plt.plot(t,evolution_overlap[n])
        temp = np.sum(evolution_overlap[n]/np.size(evolution_overlap[n]))
        temp2=temp2+temp
    plt.xlabel(r"$t$")
    plt.title(r"$N=$"+str(pxp.N)+r", $V=$"+str(V)+r", $\tau=$"+"{:.1f}".format(tau))
    plt.ylabel(r"$\vert \langle n \vert e^{i H n \tau} F^n \vert n \rangle \vert^2$")
    # plt.figure(figsize=cm2inch(21,29.7))
    # plt.get_current_fig_manager().resize(500, 200)
    plt.tight_layout()
    plt.savefig("./gif_images3/temp"+str(tau_index))
    plt.cla()
    temp2=temp2/len(evolution_overlap)
    sum_cost[tau_index] = temp2
    tau_index = tau_index+1


    # plt.xlabel(r"$t$")
    # plt.title(r"$C(t) = \vert \langle \psi(0) \vert e^{i H^{PXP} n \tau} F^n \vert \psi(0 \rangle \vert^2)$, $N=$"+str(pxp.N)+r", $\tau=$"+str(tau))
    # plt.ylabel(r"$C(t)$")
    # plt.show()
plt.plot(tau_vals,sum_cost)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$C$")
plt.title(r"Average over all basis states $\vert n \rangle$ of $\vert \langle n \vert e^{i H n \tau} F^n \vert n \rangle \vert^2$, $V=$"+str(V)+r", $N=$"+str(pxp.N))
plt.show()
