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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

def print_wf(state,system):
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>1e-5:
            print(state[n],system.basis[n])

#choose
import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#create coherent states to init trajectories from
def coherent_state(alpha,su2_states,J):
    wf_su2 = np.zeros(np.size(su2_states,axis=1),dtype=complex)

    eta = alpha*np.tan(np.abs(alpha))/np.abs(alpha)
    for m in range(0,np.size(wf_su2,axis=0)):
        wf_su2[m] = np.power(ncr(int(2*J),m),0.5)*np.power(eta,m)

    wf = wf_su2[0] * su2_states[:,0]
    for m in range(1,np.size(wf_su2,axis=0)):
        wf = wf + wf_su2[m] * su2_states[:,m]

    wf = wf / np.power(1+np.abs(eta)**2,J)
    return wf

from Calculations import time_evolve_state

def cell_no(state_energy,eigs,T,delta_t):
    t_range = np.arange(0,T,delta_t)
    lin = 1-t_range/T

    fid = np.zeros(np.size(t_range))
    for n in range(0,np.size(t_range,axis=0)):
        evolved_state = time_evolve_state(state_energy,eigs,t_range[n])
        fid[n] = np.abs(np.vdot(evolved_state,state_energy))**2
    integrand = np.multiply(lin,fid)
    integral = np.trapz(integrand,t_range)
    phase_space_cells = T/(2*integral)
    return phase_space_cells

from Calculations import time_evolve_state
def trajectory(init_state,t_range,H_eigvalues,H_eigvectors,system):
    state_energy = np.dot(np.conj(np.transpose(H_eigvectors)),init_state)
    x0 = np.vdot(init_state,np.dot(X.sector.matrix(),init_state))
    y0 = np.vdot(init_state,np.dot(Y.sector.matrix(),init_state))
    z0 = np.vdot(init_state,np.dot(Z.sector.matrix(),init_state))

    x = np.zeros(np.size(t_range))
    y = np.zeros(np.size(t_range))
    z = np.zeros(np.size(t_range))
    f = np.zeros(np.size(t_range))
    for n in range(0,np.size(t_range,axis=0)):
        evolved_state = time_evolve_state(state_energy,H_eigvalues,t_range[n])
        x[n] = np.real(np.vdot(evolved_state,np.dot(X_energy,evolved_state)))/J
        y[n] = np.real(np.vdot(evolved_state,np.dot(Y_energy,evolved_state)))/J
        z[n] = np.real(np.vdot(evolved_state,np.dot(Z_energy,evolved_state)))/J
        f[n] = np.abs(np.vdot(evolved_state,state_energy))**2
    return x,y,z,f

def project_to_subspace(state,system1,system2):
    projected_state = np.zeros(np.size(system2.basis_refs),dtype=complex)
    # projected_state = np.copy(state)
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>1e-5:
            if system1.basis_refs[n] in system2.basis_refs:
                projected_state[system2.keys[system1.basis_refs[n]]] = state[n]
            # if system1.basis_refs[n] not in system2.basis_refs:
                # projected_state[n] = 0
    projected_state = projected_state / np.power(np.vdot(projected_state,projected_state),0.5)
    return projected_state

N=18
pxp = unlocking_System([0],"periodic",2,N)
pxp_full = unlocking_System([0,1],"periodic",2,N)
pxp_syms = model_sym_data(pxp,[translational(pxp)])

# H = spin_Hamiltonian(pxp,"x",pxp_syms)
# #looking at coherent states, lin combinations of |J,m>, |J,m> = (sum_n s_n^+)^m |000....>
# #ie |J,m> lin combs of zero momenta states, only need to consider zero momenta
# H.gen()
# H.sector.find_eig()
# np.save("pxp,eigvalues,18",H.sector.eigvalues())
# np.save("pxp,eigvectors,18",H.sector.eigvectors())
H_eigvalues = np.load("./pxp,eigvalues,18.npy")
H_eigvectors = np.load("./pxp,eigvectors,18.npy")
# H_eigvalues = H.sector.eigvalues()
# H_eigvectors = H.sector.eigvectors()

# # operators for coherent states
X = spin_Hamiltonian(pxp,"x",pxp_syms)
X.gen()
Y = spin_Hamiltonian(pxp,"y",pxp_syms)
Y.gen()
Z = spin_Hamiltonian(pxp,"z",pxp_syms)
Z.gen()
sp = X.sector.matrix()+1j*Y.sector.matrix()
sm = X.sector.matrix()-1j*Y.sector.matrix()

X_energy = np.dot(np.conj(np.transpose(H_eigvectors)),np.dot(X.sector.matrix(),H_eigvectors))
Y_energy = np.dot(np.conj(np.transpose(H_eigvectors)),np.dot(Y.sector.matrix(),H_eigvectors))
Z_energy = np.dot(np.conj(np.transpose(H_eigvectors)),np.dot(Z.sector.matrix(),H_eigvectors))

#create basis of total SU(2) states
from State_Classes import total_spin_state
s=1/2*(pxp.base-1)
no_singlets=6
J = int(pxp.N*s)-no_singlets

print(J)
state = total_spin_state(J,pxp_full)
state = project_to_subspace(state,pxp_full,pxp)

su2_states = state
for n in range(0,J):
    state = np.dot(sp,state)
    state = state / np.power(np.vdot(state,state),0.5)
    su2_states = np.vstack((su2_states,state))
su2_states = np.transpose(su2_states)

# for n in range(0,np.size(su2_states,axis=1)):
    # print("State ",n)
    # print_wf(su2_states[:,n],pxp)
        
#form coherent states
states = dict()
states_energy = dict()
delta = 0.1
vals = np.arange(-math.pi,math.pi+delta,delta)
# vals = [-2.5415976535]
# vals = [0.4584073464102101]
# vals = [2.6859]
no_traj = np.size(vals)
for n in range(0,np.size(vals,axis=0)):
    a=np.random.uniform(-3,3)
    b=np.random.uniform(-3,3)
    states[n] = coherent_state(vals[n],su2_states,J)
    # states[n] = coherent_state(a+1j*b,su2_states,J)
    temp = np.abs(states[n])
    states[n] = states[n] / np.power(np.vdot(states[n],states[n]),0.5)
    states_energy[n] = np.dot(np.conj(np.transpose(H_eigvectors)),states[n])

#calc phase space accessed
# T = np.arange(0,30,0.5)
# cells = dict()
# pbar=ProgressBar()
# print("Calculating phase space cells of coherent states")
# for m in pbar(range(0,len(states_energy))):
    # cells[m] = np.zeros(np.size(T))
    # for n in range(0,np.size(T,axis=0)):
        # cells[m][n] = cell_no(states_energy[m],H_eigvalues,T[n],0.01)
# cells_gradient =dict()
# for n in range(0,len(cells)):
    # cells_gradient[n] = np.gradient(cells[n],T)

z0=zm_state(2,1,pxp)
z1=zm_state(2,1,pxp,shift=1)
z3=zm_state(3,1,pxp)
z0_energy = H_eigvectors[pxp.keys[z0.ref],:]
z1_energy = H_eigvectors[pxp.keys[z1.ref],:]
z3_energy = H_eigvectors[pxp.keys[z3.ref],:]
# k_refs=pxp_syms.find_k_ref(z0.ref)
# for n in range(1,np.size(k_refs,axis=0)):
    # H.gen(k_refs[n])
    # H.sector.find_eig(k_refs[n])
# k_z3=pxp_syms.find_k_ref(z3.ref)
# for n in range(1,np.size(k_z3,axis=0)):
    # H.gen(k_z3[n])
    # H.sector.find_eig(k_z3[n])

# #z2 state in energy basis
# #sectors init state has non zero overlap with
# k_refs = H.syms.find_k_ref(z0.ref)
# #get init state, overlap state sym representation in these sectors
# init_sym_states = dict()
# for n in range(0,np.size(k_refs,axis=0)):
    # init_sym_states[n] = z0.sym_basis(k_refs[n],H.syms)
# #take overlap with energy eigenstates to get energy rep
# init_energy_states = dict()
# for n in range(0,np.size(k_refs,axis=0)):
    # init_energy_states[n] = np.zeros(np.size(init_sym_states[n]),dtype=complex)
    # for m in range(0,np.size(init_energy_states[n],axis=0)):
        # init_energy_states[n][m] = np.conj(np.vdot(init_sym_states[n],H.sector.eigvectors(k_refs[n])[:,m]))

# #combine sector data
# z_energy = init_energy_states[0]
# e = H.sector.eigvalues(k_refs[0])
# for n in range(1,np.size(k_refs,axis=0)):
    # z_energy = np.append(z_energy,init_energy_states[n])
    # e = np.append(e,H.sector.eigvalues(k_refs[n]))

# z_cell = np.zeros(np.size(T))
# for n in range(0,np.size(T,axis=0)):
    # evolved_state = time_evolve_state(z_energy,e,T[n])
    # z_cell[n] = cell_no(z_energy,e,T[n],0.01)
# z_cell_gradient = np.gradient(z_cell,T)

# #plot cell no gradient 
# for n in range(0,len(cells)):
    # plt.plot(T,cells_gradient[n],alpha=0.5)
# # plt.plot(T,z_cell_gradient,color="blue",linewidth=3,label="Neel")
# plt.legend()
# plt.title(r"$dN(t)/dt$, Rydberg coherent states $\vert \alpha \rangle = exp(\alpha L^+ - \bar{\alpha} L^-) \vert J,0 \rangle$, PXP, N="+str(pxp.N))
# plt.xlabel(r"$t$")
# plt.ylabel(r"$N(t)$")
# plt.show()

# # #plot cell no 
# for n in range(0,len(cells)):
    # plt.plot(T,cells[n],alpha=0.5)
# # plt.plot(T,z_cell,color="blue",linewidth=3,label="Neel")
# plt.legend()
# plt.title(r"$N(t)$, Rydberg coherent states $\vert \alpha \rangle = exp(\alpha L^+ - \bar{\alpha} L^-) \vert J,0 \rangle$, PXP, N="+str(pxp.N))
# plt.xlabel(r"$t$")
# plt.ylabel(r"$N(t)$")
# plt.show()

# plot trajectories of allcoherent states
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

t_range = np.arange(0,30,0.1)
# f_neel= fidelity(z0,H).eval(t_range,z0)
# f_neel1= fidelity(z0,H).eval(t_range,z1)
# f_neel3= fidelity(z3,H).eval(t_range,z3)
f_neel = np.zeros(np.size(t_range))
f_neel1 = np.zeros(np.size(t_range))
f_neel3 = np.zeros(np.size(t_range))
for n in range(0,np.size(t_range,axis=0)):
    z2_evolved = time_evolve_state(z0_energy,H_eigvalues,t_range[n])
    z3_evolved = time_evolve_state(z3_energy,H_eigvalues,t_range[n])
    f_neel[n] = np.abs(np.vdot(z2_evolved,z0_energy))**2
    f_neel1[n] = np.abs(np.vdot(z2_evolved,z1_energy))**2
    f_neel3[n] = np.abs(np.vdot(z3_evolved,z3_energy))**2
    

pbar=ProgressBar()
print("Plotting Trajectories")
for n in pbar(range(0,len(states))):
    x,y,z,f = trajectory(states[n],t_range,H_eigvalues,H_eigvectors,pxp)
    plt.plot(t_range,f,alpha=0.5)
    np.save("./data/"+str(N)+"/fidelity/cstate_fidelity,J"+str(J)+","+str(n),f)
    # ax.plot(x,y,z)
# plt.show()

plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
plt.title("PXP, Rydberg coherent states, J="+str(J)+" Fidelity, N="+str(pxp.N))
plt.plot(t_range,f_neel,color="blue",linewidth=3,label="Neel")
plt.plot(t_range,f_neel1,color="green",linewidth=3,label="Anti-Neel")
plt.plot(t_range,f_neel3,color="magenta",linewidth=3,label="Z3")
# np.save("./data/12/fidelity/cstate_fidelity,neel",f_neel)
# np.save("./data/12/fidelity/cstate_fidelity,aneel",f_neel1)
# np.save("./data/12/fidelity/cstate_fidelity,z3",f_neel3)
plt.legend()
plt.show()

# save states from each band for further analysis
states_array=np.zeros((len(states),np.size(states[0])),dtype=complex)
for n in range(0,len(states)):
    states_array[n] = states[n]
states_array = np.transpose(states_array)
# np.save("./data/12/cstates,k0,"+str(pxp.N)+".npy",states_array)

