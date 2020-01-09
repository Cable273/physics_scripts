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
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=14
pxp = unlocking_System([0],"periodic",3,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

#orig H
Ip = Hamiltonian(pxp,pxp_syms)
Ip.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip.site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip.model = np.array([[0,1,0],[0,2,0]])
Ip.model_coef = np.array([1,-1])
Ip.uc_size = np.array([2,2])
Ip.uc_pos = np.array([1,0])

Kp = Hamiltonian(pxp,pxp_syms)
Kp.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp.model = np.array([[0,1,0],[0,2,0]])
Kp.model_coef = np.array([1,-1])
Kp.uc_size = np.array([2,2])
Kp.uc_pos = np.array([1,0])

Lp = Hamiltonian(pxp,pxp_syms)
Lp.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp.model = np.array([[0,1,0],[0,2,0]])
Lp.model_coef = np.array([1,-1])
Lp.uc_size = np.array([2,2])
Lp.uc_pos = np.array([1,0])

#pertubations
Ip_pert = Hamiltonian(pxp,pxp_syms)
Ip_pert.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip_pert.site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Ip_pert.model_coef = np.array([1,1,-1,-1])
Ip_pert.uc_size = np.array([2,2,2,2])
Ip_pert.uc_pos = np.array([0,1,1,0])

Kp_pert = Hamiltonian(pxp,pxp_syms)
Kp_pert.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Kp_pert.model_coef = np.array([1,1,-1,-1])
Kp_pert.uc_size = np.array([2,2,2,2])
Kp_pert.uc_pos = np.array([0,1,1,0])

Lp_pert = Hamiltonian(pxp,pxp_syms)
Lp_pert.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Lp_pert.model_coef = np.array([1,1,-1,-1])
Lp_pert.model_coef = np.array([0,0,0,0])
# Lp_pert.uc_size = np.array([2,2,2,2])
Lp_pert.uc_pos = np.array([0,1,1,0])

z=zm_state(2,1,pxp)
# Ip.gen()
# Im.gen()
# Kp.gen()
# Km.gen()
# Lp.gen()
# Lm.gen()
# Ip_pert.gen()
# Im_pert.gen()
# Kp_pert.gen()
# Km_pert.gen()
# Lp_pert.gen()
# Lm_pert.gen()

k=[0,0]
Ip.gen(k)
Im = Ip.herm_conj()
Kp.gen(k)
Km = Kp.herm_conj()
Lp.gen(k)
Lm = Lp.herm_conj()
Ip_pert.gen(k)
Im_pert = Ip_pert.herm_conj()
Kp_pert.gen(k)
Km_pert = Kp_pert.herm_conj()
Lp_pert.gen(k)
Lm_pert = Lp_pert.herm_conj()

def gen_su3Basis(coef):
    Ip_total = H_operations.add(Ip,Ip_pert,np.array([1,coef]))
    Im_total = H_operations.add(Im,Im_pert,np.array([1,coef]))
    Kp_total = H_operations.add(Kp,Kp_pert,np.array([1,coef]))
    Km_total = H_operations.add(Km,Km_pert,np.array([1,coef]))
    Lp_total = H_operations.add(Lp,Lp_pert,np.array([1,coef]))
    Lm_total = H_operations.add(Lm,Lm_pert,np.array([1,coef]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))

    #su3 rep
    #0th order rep (no perts)
    root3 = np.power(3,0.5)
    I3 = 1/2 * com(Ip.sector.matrix(k),Im.sector.matrix(k))
    g8 = 1/(2*root3) * ( com(Kp.sector.matrix(k),Km.sector.matrix(k)) + com(Lp.sector.matrix(k),Lm.sector.matrix(k)) )

    def exp(Q,psi):
        return np.real(np.vdot(psi,np.dot(Q,psi)))
    def var(Q,psi):
        Q2 = np.dot(Q,Q)
        return exp(Q2,psi)-(exp(Q,psi))**2

    e,u = np.linalg.eigh(I3)
    lw = u[:,0]
    hw = u[:,np.size(u,axis=1)-1]
    #generate su3 representation by applying I+,L- to lw state
    su3_basis_states = dict()
    su3_basis = lw
    current_state = su3_basis
    i3_lw = exp(I3,lw)
    g8_hw = exp(g8,lw)
    current_S = np.abs(i3_lw)
    Ip_app= 0
    Lm_app= 0
    application_index = np.zeros(2)
    while np.abs(current_S)>1e-5:
        no_ip_apps = int(2*current_S)
        tagged_state = current_state
        for n in range(0,no_ip_apps):
            Ip_app = Ip_app + 1
            next_state = np.dot(Ip.sector.matrix(k),current_state)
            next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
            su3_basis = np.vstack((su3_basis,next_state))
            current_state = next_state
            application_index = np.vstack((application_index,np.array([Ip_app,Lm_app])))
        current_state = np.dot(Lm.sector.matrix(k),tagged_state)
        current_state = current_state / np.power(np.vdot(current_state,current_state),0.5)
        su3_basis = np.vstack((su3_basis,current_state))
        current_S = current_S - 1/2
        Ip_app = 0
        Lm_app += 1
        application_index = np.vstack((application_index,np.array([Ip_app,Lm_app])))
    su3_basis = np.transpose(su3_basis)

    #generate su3 dual representation (starting from highest weight state) representation by applying I-,L+ to lw state
    su3_basis_dual = hw
    current_state = su3_basis_dual
    i3_hw = exp(I3,hw)
    g8_lw = exp(g8,hw)
    current_S = np.abs(i3_hw)
    Im_app= 0
    Lp_app= 0
    application_index_dual = np.zeros(2)
    while np.abs(current_S)>1e-5:
        no_im_apps = int(2*current_S)
        tagged_state = current_state
        for n in range(0,no_im_apps):
            Im_app = Im_app + 1
            next_state = np.dot(Im.sector.matrix(k),current_state)
            next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
            su3_basis_dual = np.vstack((su3_basis_dual,next_state))
            current_state = next_state
            application_index_dual = np.vstack((application_index_dual,np.array([Im_app,Lp_app])))
        current_state = np.dot(Lp.sector.matrix(k),tagged_state)
        current_state = current_state / np.power(np.vdot(current_state,current_state),0.5)
        su3_basis_dual = np.vstack((su3_basis_dual,current_state))
        current_S = current_S - 1/2
        Im_app = 0
        Lp_app += 1
        application_index_dual = np.vstack((application_index_dual,np.array([Im_app,Lp_app])))
    su3_basis_dual = np.transpose(su3_basis_dual)

    su3_basis = np.hstack((su3_basis,su3_basis_dual))
    from Calculations import gram_schmidt
    gs = gram_schmidt(su3_basis)
    gs.ortho()
    su3_basis = gs.ortho_basis
    return su3_basis

def subspace_variance(coef):
    coef = coef[0]
    Ip_total = H_operations.add(Ip,Ip_pert,np.array([1,coef]))
    Im_total = H_operations.add(Im,Im_pert,np.array([1,coef]))
    Kp_total = H_operations.add(Kp,Kp_pert,np.array([1,coef]))
    Km_total = H_operations.add(Km,Km_pert,np.array([1,coef]))
    Lp_total = H_operations.add(Lp,Lp_pert,np.array([1,coef]))
    Lm_total = H_operations.add(Lm,Lm_pert,np.array([1,coef]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))
    su3_basis = gen_su3Basis(coef)

    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H2,su3_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)
    return subspace_variance/np.size(su3_basis,axis=1)

def subspace_varianceSu2(coef):
    coef = coef[0]
    Ip_total = H_operations.add(Ip,Ip_pert,np.array([1,coef]))
    Im_total = H_operations.add(Im,Im_pert,np.array([1,coef]))
    Kp_total = H_operations.add(Kp,Kp_pert,np.array([1,coef]))
    Km_total = H_operations.add(Km,Km_pert,np.array([1,coef]))
    Lp_total = H_operations.add(Lp,Lp_pert,np.array([1,coef]))
    Lm_total = H_operations.add(Lm,Lm_pert,np.array([1,coef]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))
    su3_basis = gen_su3Basis(coef)

    # restrict to 2N+1 basis with largest overlap with scar states
    # identify 2N+1 scars from H
    H.sector.find_eig(k)
    overlap = eig_overlap(z,H,k).eval()
    from Calculations import get_top_band_indices
    scar_indices = get_top_band_indices(H.sector.eigvalues(k),overlap,2*pxp.N,100,200,e_diff=0.5)
    plt.scatter(H.sector.eigvalues(k),overlap)
    for n in range(0,np.size(scar_indices,axis=0)):
        plt.scatter(H.sector.eigvalues(k)[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
    plt.show()
        

    #redifine su3_basis as the ritz vectors (new linear combs)
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    e,u = np.linalg.eigh(H_fsa)
    su3_basis = np.dot(su3_basis,u)

    #find 2N+1 basis states with largest overlap with scars states
    max_scar_overlap = np.zeros(np.size(su3_basis,axis=1))
    for n in range(0,np.size(max_scar_overlap,axis=0)):
        scarOverlap = np.zeros(np.size(scar_indices))
        for m in range(0,np.size(scarOverlap,axis=0)):
            scarOverlap[m] = np.vdot(su3_basis[:,n],H.sector.eigvectors(k)[:,scar_indices[m]])
        max_scar_overlap[n] = np.max(scarOverlap)

    su3_indices = np.arange(0,np.size(su3_basis,axis=1))
    max_scar_overlap,su3_indices = (list(t) for t in zip(*sorted(zip(max_scar_overlap,su3_indices))))
    max_scar_overlap = np.flip(max_scar_overlap)
    su3_indices = np.flip(su3_indices)
    su3_sub_indices = su3_indices[:np.size(scar_indices)]

    su3_sub_basis = np.zeros(np.size(su3_basis,axis=0))
    for n in range(0,np.size(su3_sub_indices,axis=0)):
        su3_sub_basis = np.vstack((su3_sub_basis,su3_basis[:,su3_sub_indices[n]]))
    su3_sub_basis = np.transpose(np.delete(su3_sub_basis,0,axis=0))
        
    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H2,su3_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)

    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_sub_basis)),np.dot(H2,su3_sub_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_sub_basis)),np.dot(H.sector.matrix(k),su3_sub_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)
    return subspace_variance/np.size(su3_sub_basis,axis=1)

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return f

from scipy.optimize import minimize_scalar
def fidelity_erorr(coef):
    coef = coef[0]
    Ip_total = H_operations.add(Ip,Ip_pert,np.array([1,coef]))
    Im_total = H_operations.add(Im,Im_pert,np.array([1,coef]))
    Kp_total = H_operations.add(Kp,Kp_pert,np.array([1,coef]))
    Km_total = H_operations.add(Km,Km_pert,np.array([1,coef]))
    Lp_total = H_operations.add(Lp,Lp_pert,np.array([1,coef]))
    Lm_total = H_operations.add(Lm,Lm_pert,np.array([1,coef]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))

    H.sector.find_eig(k)

    z=zm_state(2,1,pxp,1)
    block_refs = pxp_syms.find_block_refs(k)
    psi = np.zeros(np.size(block_refs))
    loc = find_index_bisection(z.ref,block_refs)
    psi[loc] = 1
    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),psi)

    t=np.arange(0,20,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(k),t[n])
        f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f0 = np.max(f[cut:])
    plt.scatter(H.sector.eigvalues(k),np.log10(np.abs(psi_energy)**2))
    plt.title(r"$PCP+\lambda(PPCP+PCPP), N=$"+str(pxp.N))
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
    plt.show()
    plt.plot(t,f)
    plt.title(r"$PCP+\lambda(PPCP+PCPP), N=$"+str(pxp.N))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
    plt.show()
    return 1-f0
    
# coef = 0
# from scipy.optimize import minimize
# res = minimize(lambda coef: subspace_variance(coef),method="powell",x0=[coef])
# plt.cla()
# coef = res.x
# np.save("pcp,1stOrderSubVar,coef,"+str(pxp.N),coef)
# print(res.x)
# subspace_variance([res.x])
# fidelity_erorr([res.x])

# subspace_varianceSu2([0])

coef = 0
errors0 = np.zeros(3)
errors0[0]= fidelity_erorr([coef])
errors0[1]= subspace_variance([coef])
errors0[2]= subspace_varianceSu2([coef])
np.save("pcp,0thOrderErrors,"+str(pxp.N),errors0)

errors = np.zeros((2,3))
coef = -0.09443
errors[0,0]= fidelity_erorr([coef])
errors[0,1]= subspace_variance([coef])
errors[0,2]= subspace_varianceSu2([coef])

coef = np.load("./data/1stOrderSubVar/single/pcp,1stOrderSubVar,coef,12.npy")
print(coef)
errors[1,0]= fidelity_erorr([coef])
errors[1,1]= subspace_variance([coef])
errors[1,2]= subspace_varianceSu2([coef])

np.save("pcp,1stOrderErrors,"+str(pxp.N),errors)

print(errors0)
print(errors)
