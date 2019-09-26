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


from State_Classes import coherent_state
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
def trajectory(init_state,t_range,H,system):
    state_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),init_state)
    x0 = np.vdot(init_state,np.dot(X.sector.matrix(k),init_state))
    y0 = np.vdot(init_state,np.dot(Y.sector.matrix(k),init_state))
    z0 = np.vdot(init_state,np.dot(Z.sector.matrix(k),init_state))

    x = np.zeros(np.size(t_range))
    y = np.zeros(np.size(t_range))
    z = np.zeros(np.size(t_range))
    f = np.zeros(np.size(t_range))
    for n in range(0,np.size(t_range,axis=0)):
        evolved_state = time_evolve_state(state_energy,H.sector.eigvalues(k),t_range[n])
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
        

pxp = unlocking_System([0],"periodic",2,12)
pxp_full = unlocking_System([0,1],"periodic",2,12)
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
#looking at coherent states, lin combinations of |J,m>, |J,m> = (sum_n s_n^+)^m |000....>
#ie |J,m> lin combs of zero momenta states, only need to consider zero momenta
k=[0]
H.gen(k)
H.sector.find_eig(k)
mom_refs = pxp_syms.find_block_refs(k)
mom_keys = dict()
for n in range(0,np.size(mom_refs,axis=0)):
    mom_keys[mom_refs[n]] = n

# operators for coherent states
X = spin_Hamiltonian(pxp,"x",pxp_syms)
X.gen(k)
Y = spin_Hamiltonian(pxp,"y",pxp_syms)
Y.gen(k)
Z = spin_Hamiltonian(pxp,"z",pxp_syms)
Z.gen(k)
sp = X.sector.matrix(k)+1j*Y.sector.matrix(k)
sm = X.sector.matrix(k)-1j*Y.sector.matrix(k)
spe,spu = np.linalg.eig(sp)
sme,smu = np.linalg.eig(sm)

X_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(X.sector.matrix(k),H.sector.eigvectors(k)))
Y_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(Y.sector.matrix(k),H.sector.eigvectors(k)))
Z_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),np.dot(Z.sector.matrix(k),H.sector.eigvectors(k)))

#create basis of total SU(2) states
from State_Classes import total_spin_state
s=1/2*(pxp.base-1)
J = int(pxp.N*s)
state = total_spin_state(J,pxp_full)
state = project_to_subspace(state,pxp_full,pxp)
print(np.size(state))
su2_states = state
for n in range(0,J):
    state = np.dot(sp,state)
    state = state / np.power(np.vdot(state,state),0.5)
    su2_states = np.vstack((su2_states,state))
su2_states = np.transpose(su2_states)

for n in range(0,np.size(su2_states,axis=1)):
    print_wf(su2_states[:,n])
        

#form coherent states
# states = dict()
# states_energy = dict()
# delta = 0.5
# vals = np.arange(-math.pi,math.pi+delta,delta)
# # vals = [-2.5415976535]
# # vals = [0.4584073464102101]
# # vals = [2.6859]
# for n in range(0,np.size(vals,axis=0)):
    # states[n] = coherent_state(vals[n],su2_states,J)
    # temp = np.abs(states[n])
    # states[n] = states[n] / np.power(np.vdot(states[n],states[n]),0.5)
    # states_energy[n] = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),states[n])

# #calc phase space accessed
# T = np.arange(0,30,0.5)
# cells = dict()
# pbar=ProgressBar()
# print("Calculating phase space cells of coherent states")
# for m in pbar(range(0,len(states_energy))):
    # cells[m] = np.zeros(np.size(T))
    # for n in range(0,np.size(T,axis=0)):
        # cells[m][n] = cell_no(states_energy[m],H.sector.eigvalues(k),T[n],0.01)
# cells_gradient =dict()
# for n in range(0,len(cells)):
    # cells_gradient[n] = np.gradient(cells[n],T)

# z0=zm_state(2,1,pxp)
# z1=zm_state(2,1,pxp,shift=1)
# k_refs=pxp_syms.find_k_ref(z0.ref)
# for n in range(1,np.size(k_refs,axis=0)):
    # H.gen(k_refs[n])
    # H.sector.find_eig(k_refs[n])
# z3=zm_state(3,1,pxp)
# k_z3=pxp_syms.find_k_ref(z3.ref)
# for n in range(1,np.size(k_z3,axis=0)):
    # H.gen(k_z3[n])
    # H.sector.find_eig(k_z3[n])

# # #z2 state in energy basis
# # #sectors init state has non zero overlap with
# # k_refs = H.syms.find_k_ref(z0.ref)
# # #get init state, overlap state sym representation in these sectors
# # init_sym_states = dict()
# # for n in range(0,np.size(k_refs,axis=0)):
    # # init_sym_states[n] = z0.sym_basis(k_refs[n],H.syms)
# # #take overlap with energy eigenstates to get energy rep
# # init_energy_states = dict()
# # for n in range(0,np.size(k_refs,axis=0)):
    # # init_energy_states[n] = np.zeros(np.size(init_sym_states[n]),dtype=complex)
    # # for m in range(0,np.size(init_energy_states[n],axis=0)):
        # # init_energy_states[n][m] = np.conj(np.vdot(init_sym_states[n],H.sector.eigvectors(k_refs[n])[:,m]))

# # #combine sector data
# # z_energy = init_energy_states[0]
# # e = H.sector.eigvalues(k_refs[0])
# # for n in range(1,np.size(k_refs,axis=0)):
    # # z_energy = np.append(z_energy,init_energy_states[n])
    # # e = np.append(e,H.sector.eigvalues(k_refs[n]))

# # z_cell = np.zeros(np.size(T))
# # for n in range(0,np.size(T,axis=0)):
    # # evolved_state = time_evolve_state(z_energy,e,T[n])
    # # z_cell[n] = cell_no(z_energy,e,T[n],0.01)
# # z_cell_gradient = np.gradient(z_cell,T)

# # #plot cell no gradient 
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

# # plot trajectories of allcoherent states
# # import matplotlib as mpl
# # from mpl_toolkits.mplot3d import Axes3D
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")

# t_range = np.arange(0,20,0.01)
# f_neel= fidelity(z0,H,"use sym").eval(t_range,z0)
# f_neel1= fidelity(z0,H,"use sym").eval(t_range,z1)
# f_neel3= fidelity(z3,H,"use sym").eval(t_range,z3)

# pbar=ProgressBar()
# print("Plotting Trajectories")
# for n in pbar(range(0,len(states))):
    # x,y,z,f = trajectory(states[n],t_range,H,pxp)
    # plt.plot(t_range,f,alpha=0.5)
    # # np.save("./data/12/fidelity/cstate_fidelity,"+str(n),f)
    # # ax.plot(x,y,z)

# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
# plt.title("PXP, Rydberg coherent states, Fidelity, N="+str(pxp.N))
# plt.plot(t_range,f_neel,color="blue",linewidth=3,label="Neel")
# plt.plot(t_range,f_neel1,color="green",linewidth=3,label="Anti-Neel")
# plt.plot(t_range,f_neel3,color="magenta",linewidth=3,label="Z3")
# # np.save("./data/12/fidelity/cstate_fidelity,neel",f_neel)
# # np.save("./data/12/fidelity/cstate_fidelity,aneel",f_neel1)
# # np.save("./data/12/fidelity/cstate_fidelity,z3",f_neel3)
# plt.legend()
# plt.show()

# # save states from each band for further analysis
# states_array=np.zeros((len(states),np.size(states[0])),dtype=complex)
# for n in range(0,len(states)):
    # states_array[n] = states[n]
# states_array = np.transpose(states_array)
# # np.save("./data/12/cstates,k0,"+str(pxp.N)+".npy",states_array)

