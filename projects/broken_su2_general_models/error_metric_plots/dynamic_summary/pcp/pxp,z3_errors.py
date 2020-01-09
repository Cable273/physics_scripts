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
N=10
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

Im = Hamiltonian(pxp,pxp_syms)
Im.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im.site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im.model = np.array([[0,2,0],[0,1,0]])
Im.model_coef = np.array([1,-1])
Im.uc_size = np.array([2,2])
Im.uc_pos = np.array([1,0])

Kp = Hamiltonian(pxp,pxp_syms)
Kp.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp.model = np.array([[0,1,0],[0,2,0]])
Kp.model_coef = np.array([1,-1])
Kp.uc_size = np.array([2,2])
Kp.uc_pos = np.array([1,0])

Km = Hamiltonian(pxp,pxp_syms)
Km.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km.model = np.array([[0,2,0],[0,1,0]])
Km.model_coef = np.array([1,-1])
Km.uc_size = np.array([2,2])
Km.uc_pos = np.array([1,0])

Lp = Hamiltonian(pxp,pxp_syms)
Lp.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp.model = np.array([[0,1,0],[0,2,0]])
Lp.model_coef = np.array([1,-1])
Lp.uc_size = np.array([2,2])
Lp.uc_pos = np.array([1,0])

Lm = Hamiltonian(pxp,pxp_syms)
Lm.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lm.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lm.model = np.array([[0,2,0],[0,1,0]])
Lm.model_coef = np.array([1,-1])
Lm.uc_size = np.array([2,2])
Lm.uc_pos = np.array([1,0])

#pertubations
Ip_pert = Hamiltonian(pxp,pxp_syms)
Ip_pert.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip_pert.site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Ip_pert.model_coef = np.array([1,1,-1,-1])
Ip_pert.uc_size = np.array([2,2,2,2])
Ip_pert.uc_pos = np.array([0,1,1,0])

Im_pert = Hamiltonian(pxp,pxp_syms)
Im_pert.site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im_pert.site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im_pert.model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Im_pert.model_coef = np.array([1,1,-1,-1])
Im_pert.uc_size = np.array([2,2,2,2])
Im_pert.uc_pos = np.array([0,1,1,0])

Kp_pert = Hamiltonian(pxp,pxp_syms)
Kp_pert.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Kp_pert.model_coef = np.array([1,1,-1,-1])
Kp_pert.uc_size = np.array([2,2,2,2])
Kp_pert.uc_pos = np.array([0,1,1,0])

Km_pert = Hamiltonian(pxp,pxp_syms)
Km_pert.site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km_pert.model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Km_pert.model_coef = np.array([1,1,-1,-1])
Km_pert.uc_size = np.array([2,2,2,2])
Km_pert.uc_pos = np.array([0,1,1,0])

Lp_pert = Hamiltonian(pxp,pxp_syms)
Lp_pert.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp_pert.model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Lp_pert.model_coef = np.array([1,1,-1,-1])
Lp_pert.uc_size = np.array([2,2,2,2])
Lp_pert.uc_pos = np.array([0,1,1,0])

Lm_pert = Hamiltonian(pxp,pxp_syms)
Lm_pert.site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lm_pert.site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lm_pert.model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Lm_pert.model_coef = np.array([1,1,-1,-1])
Lm_pert.uc_size = np.array([2,2,2,2])
Lm_pert.uc_pos = np.array([0,1,1,0])

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

z=zm_state(2,1,pxp)
# k=pxp_syms.find_k_ref(z.ref)
k=[0,0]
Ip.gen(k)
Im.gen(k)
Kp.gen(k)
Km.gen(k)
Lp.gen(k)
Lm.gen(k)
Ip_pert.gen(k)
Im_pert.gen(k)
Kp_pert.gen(k)
Km_pert.gen(k)
Lp_pert.gen(k)
Lm_pert.gen(k)

coef = 0
# coef = -0.0827840215
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
exact_energy = H.sector.eigvalues(k)
exact_overlap= eig_overlap(z,H,k).eval()

plt.scatter(exact_energy,exact_overlap)
plt.show()
t=np.arange(0,20,0.01)
f=fidelity(z,H,"use sym").eval(t,z)
plt.plot(t,f)
plt.show()
    
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
print(current_S)
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

I3 = 1/2 * com(Ip_total.sector.matrix(k),Im_total.sector.matrix(k))
root3=np.power(3,0.5)
g8 = 1/(2*root3)*( com(Kp_total.sector.matrix(k),Km_total.sector.matrix(k)) + com(Lp_total.sector.matrix(k),Lm_total.sector.matrix(k)) )
e,u = np.linalg.eigh(I3)
lw = u[:,0]

#generate new su3 rep
su3_basis= lw
for n in range(0,np.size(application_index,axis=0)):
    temp_state = lw
    for u in range(0,int(application_index[n,1])):
        temp_state = np.dot(Lm.sector.matrix(k),temp_state)
    for u in range(0,int(application_index[n,0])):
        temp_state = np.dot(Ip.sector.matrix(k),temp_state)
    temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    su3_basis = np.vstack((su3_basis,temp_state))
su3_basis = np.transpose(su3_basis)

#generate new su3 rep dual
su3_basis_dual= hw
for n in range(0,np.size(application_index_dual,axis=0)):
    temp_state = hw
    for u in range(0,int(application_index_dual[n,1])):
        temp_state = np.dot(Lp.sector.matrix(k),temp_state)
    for u in range(0,int(application_index_dual[n,0])):
        temp_state = np.dot(Im_total.sector.matrix(k),temp_state)
    temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    su3_basis_dual = np.vstack((su3_basis_dual,temp_state))
su3_basis_dual = np.transpose(su3_basis_dual)

su3_basis = np.hstack((su3_basis,su3_basis_dual))
from Calculations import gram_schmidt
gs = gram_schmidt(su3_basis)
gs.ortho()
su3_basis = gs.ortho_basis

H_su3 = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
e,u = np.linalg.eigh(H_su3)
fsa_energy = e
fsa_overlap = np.log10(np.abs(u[0,:])**2)
plt.scatter(exact_energy,exact_overlap)
plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
plt.show()

# np.save("pcp,0th_order,e,"+str(pxp.N),exact_energy)
# np.save("pcp,0th_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pcp,0th_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pcp,z2_fsa,0th_order,e,"+str(pxp.N),fsa_energy)
# np.save("pcp,z2_fsa,0th_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pcp,1st_order,e,"+str(pxp.N),exact_energy)
# np.save("pcp,1st_order,z2_overlap,"+str(pxp.N),exact_overlap)
# np.save("pcp,1st_order,z2_fidelity,"+str(pxp.N),f)
# np.save("pcp,z2_fsa,1st_order,e,"+str(pxp.N),fsa_energy)
# np.save("pcp,z2_fsa,1st_order,z2_overlap,"+str(pxp.N),fsa_overlap)

# np.save("pxp,2nd_order,e,"+str(pxp.N),exact_energy)
# np.save("pxp,2nd_order,z3_overlap,"+str(pxp.N),exact_overlap)
# # np.save("pxp,2nd_order,z3_fidelity,"+str(pxp.N),f)
# np.save("pxp,z3_fsa,2nd_order,e,"+str(pxp.N),fsa_energy)
# np.save("pxp,z3_fsa,2nd_order,z3_overlap,"+str(pxp.N),fsa_overlap)
