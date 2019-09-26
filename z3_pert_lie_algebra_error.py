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
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 6
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=3)])
z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,0,0]])
V1_ops[0].model_coef = np.array([1])
V1_ops[0].gen(uc_size=3,uc_pos=1)

V1_ops[1] = Hamiltonian(pxp,pxp_syms)
V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[1].model = np.array([[0,0,1,0]])
V1_ops[1].model_coef = np.array([1])
V1_ops[1].gen(uc_size=3,uc_pos=2)

V1_ops[2] = Hamiltonian(pxp,pxp_syms)
V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[2].model = np.array([[0,1,0,0]])
V1_ops[2].model_coef = np.array([1])
V1_ops[2].gen(uc_size=3,uc_pos=2)

V1_ops[3] = Hamiltonian(pxp,pxp_syms)
V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[3].model = np.array([[0,0,1,0]])
V1_ops[3].model_coef = np.array([1])
V1_ops[3].gen(uc_size=3,uc_pos=1)

V1 = V1_ops[0]
for n in range(1,len(V1_ops)):
    V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,0,1,0]])
V2_ops[0].model_coef = np.array([1])
V2_ops[0].gen(uc_size=3,uc_pos=0)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,0,0]])
V2_ops[1].model_coef = np.array([1])
V2_ops[1].gen(uc_size=3,uc_pos=0)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,1,1,1,0]])
V3_ops[0].model_coef = np.array([1])
V3_ops[0].gen(uc_size=3,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,1,1,0]])
V3_ops[1].model_coef = np.array([1])
V3_ops[1].gen(uc_size=3,uc_pos=2)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()

Hp_ops = dict()
Hp_ops[0] = Hamiltonian(pxp,pxp_syms)
Hp_ops[0].site_ops[1] = np.array([[0,1],[0,0]])
Hp_ops[0].model = np.array([[0,1,0]])
Hp_ops[0].model_coef = np.array([1])
Hp_ops[0].gen(uc_size=3,uc_pos=2)
Hp_ops[1] = Hamiltonian(pxp,pxp_syms)
Hp_ops[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[1].model = np.array([[0,1,0]])
Hp_ops[1].model_coef = np.array([1])
Hp_ops[1].gen(uc_size=3,uc_pos=0)
Hp_ops[2] = Hamiltonian(pxp,pxp_syms)
Hp_ops[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[2].model = np.array([[0,1,0]])
Hp_ops[2].model_coef = np.array([1])
Hp_ops[2].gen(uc_size=3,uc_pos=1)

Hp_ops[3] = Hamiltonian(pxp,pxp_syms)
Hp_ops[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[3].model = np.array([[0,0,1,0]])
Hp_ops[3].model_coef = np.array([1])
Hp_ops[3].gen(uc_size=3,uc_pos=2)
Hp_ops[4] = Hamiltonian(pxp,pxp_syms)
Hp_ops[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[4].model = np.array([[0,1,0,0]])
Hp_ops[4].model_coef = np.array([1])
Hp_ops[4].gen(uc_size=3,uc_pos=1)
Hp_ops[5] = Hamiltonian(pxp,pxp_syms)
Hp_ops[5].site_ops[1] = np.array([[0,1],[0,0]])
Hp_ops[5].model = np.array([[0,1,0,0]])
Hp_ops[5].model_coef = np.array([1])
Hp_ops[5].gen(uc_size=3,uc_pos=2)
Hp_ops[6] = Hamiltonian(pxp,pxp_syms)
Hp_ops[6].site_ops[1] = np.array([[0,1],[0,0]])
Hp_ops[6].model = np.array([[0,0,1,0]])
Hp_ops[6].model_coef = np.array([1])
Hp_ops[6].gen(uc_size=3,uc_pos=1)

#pp+p + p+pp 1st p on site 3n
Hp_ops[7] = Hamiltonian(pxp,pxp_syms)
Hp_ops[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[7].model = np.array([[0,1,0,0]])
Hp_ops[7].model_coef = np.array([1])
Hp_ops[7].gen(uc_size=3,uc_pos=0)
Hp_ops[8] = Hamiltonian(pxp,pxp_syms)
Hp_ops[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[8].model = np.array([[0,0,1,0]])
Hp_ops[8].model_coef = np.array([1])
Hp_ops[8].gen(uc_size=3,uc_pos=0)

#P-+-P terms
Hp_ops[9] = Hamiltonian(pxp,pxp_syms)
Hp_ops[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp_ops[9].model = np.array([[0,2,1,2,0]])
Hp_ops[9].model_coef = np.array([1])
Hp_ops[9].gen(uc_size=3,uc_pos=2)
Hp_ops[10] = Hamiltonian(pxp,pxp_syms)
Hp_ops[10].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[10].site_ops[2] = np.array([[0,1],[0,0]])
Hp_ops[10].model = np.array([[0,2,1,2,0]])
Hp_ops[10].model_coef = np.array([1])
Hp_ops[10].gen(uc_size=3,uc_pos=0)

def F_norm(A):
    return np.power(np.trace(np.dot(A,np.conj(np.transpose(A)))),0.5)

def lie_algebra_error(coef,plot=False):
    if plot == True:
        H = H_operations.add(H0,V1,np.array([1,coef[0]]))
        H = H_operations.add(H,V2,np.array([1,coef[1]]))
        H = H_operations.add(H,V3,np.array([1,coef[2]]))
        H.sector.find_eig()
        z=zm_state(3,1,pxp)
        fidelity(z,H).plot(np.arange(0,20,0.01),z)
        plt.show()

    Hp = Hp_ops[0]
    Hp = H_operations.add(Hp,Hp_ops[1],np.array([1,1]))
    Hp = H_operations.add(Hp,Hp_ops[2],np.array([1,1]))

    Hp = H_operations.add(Hp,Hp_ops[3],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[4],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[5],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[6],np.array([1,coef[0]]))

    Hp = H_operations.add(Hp,Hp_ops[7],np.array([1,coef[1]]))
    Hp = H_operations.add(Hp,Hp_ops[8],np.array([1,coef[1]]))

    Hp = H_operations.add(Hp,Hp_ops[9],np.array([1,coef[2]]))
    Hp = H_operations.add(Hp,Hp_ops[10],np.array([1,coef[2]]))


    Hp = Hp.sector.matrix()
    Hm = np.conj(np.transpose(Hp))
    # temp = Hp + Hm
    # print((np.abs(temp-H.sector.matrix())<1e-5).all())
    def com(a,b):
        return np.dot(a,b)-np.dot(b,a)
    Hz = 1/2 * com(Hp,Hm)
    lie_Hp = com(Hz,Hp)
    lie_Hm = com(Hz,Hm)
    diff_Hp = lie_Hp - Hp
    diff_Hm = lie_Hm + Hm
    error = F_norm(diff_Hp)
    # print(coef,error)
    return error

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f= np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

def fidelity_error(coef,plot=False):
    H = H_operations.add(H0,V1,np.array([1,coef[0]]))
    H = H_operations.add(H,V2,np.array([1,coef[1]]))
    H = H_operations.add(H,V3,np.array([1,coef[2]]))
    H.sector.find_eig()
    z=zm_state(3,1,pxp)

    if plot == True:
        fidelity(z,H).plot(np.arange(0,20,0.01),z)
        plt.show()

    psi_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t:fidelity_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(3.0,4.0))
    f_max = fidelity_eval(psi_energy,H.sector.eigvalues(),res.x)
    # print(coef,res.x,f_max)
    if res.x<1e-5:
        return 1000
    else:
        return f_max

def z_var_error(coef,plot=False):
    # H = H_operations.add(H0,V1,np.array([1,coef[0]]))
    # H = H_operations.add(H,V2,np.array([1,coef[1]]))
    # H = H_operations.add(H,V3,np.array([1,coef[2]]))

    Hp = Hp_ops[0]
    Hp = H_operations.add(Hp,Hp_ops[1],np.array([1,1]))
    Hp = H_operations.add(Hp,Hp_ops[2],np.array([1,1]))

    Hp = H_operations.add(Hp,Hp_ops[3],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[4],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[5],np.array([1,coef[0]]))
    Hp = H_operations.add(Hp,Hp_ops[6],np.array([1,coef[0]]))

    Hp = H_operations.add(Hp,Hp_ops[7],np.array([1,coef[1]]))
    Hp = H_operations.add(Hp,Hp_ops[8],np.array([1,coef[1]]))

    Hp = H_operations.add(Hp,Hp_ops[9],np.array([1,coef[2]]))
    Hp = H_operations.add(Hp,Hp_ops[10],np.array([1,coef[2]]))


    Hp = Hp.sector.matrix()
    Hm = np.conj(np.transpose(Hp))
    def com(a,b):
        return np.dot(a,b)-np.dot(b,a)
    Hz = 1/2*com(Hp,Hm)
    Hz2 = np.dot(Hz,Hz)

    fsa_basis = zm_state(3,1,pxp).prod_basis()
    current_state = fsa_basis
    fsa_dim = int(2*pxp.N/3)
    for n in range(0,fsa_dim):
        next_state = np.dot(Hp,current_state)
        next_state = next_state  / np.power(np.vdot(next_state,next_state),0.5)
        fsa_basis = np.vstack((fsa_basis,next_state))
        current_state = next_state
    fsa_basis = np.transpose(fsa_basis)

    #error = sum [ var(fsa_basis) ]
    error = 0
    for n in range(0,np.size(fsa_basis,axis=1)):
        exp_Hz_2  = np.vdot(fsa_basis[:,n],np.dot(Hz2,fsa_basis[:,n]))
        exp_Hz  = np.vdot(fsa_basis[:,n],np.dot(Hz,fsa_basis[:,n]))
        error += exp_Hz_2 - np.power(exp_Hz,2)
    # print(coef,error)
    return error

d=0.01
coef_range = np.arange(-0.2,0.2+d,d)
M = np.zeros((np.size(coef_range),np.size(coef_range)))
M2 = np.zeros((np.size(coef_range),np.size(coef_range)))
M3 = np.zeros((np.size(coef_range),np.size(coef_range)))
pbar=ProgressBar()
for n in pbar(range(0,np.size(coef_range,axis=0))):
    for m in range(0,np.size(coef_range,axis=0)):
        M[n,m] = lie_algebra_error(np.array([coef_range[n],coef_range[m],0]))
        M2[n,m] = z_var_error(np.array([coef_range[n],coef_range[m],0]))
        M3[n,m] = -fidelity_error(np.array([coef_range[n],coef_range[m],0]))
x,y = np.meshgrid(coef_range,coef_range)

# plt.contourf(x,y,M,levels=np.arange(0,10,0.2))
# plt.colorbar()
# plt.scatter(-0.170039,0.099826,marker="x",color="red",s=100)
# plt.xlabel(r"$\lambda_2$")
# plt.ylabel(r"$\lambda_1$",rotation=0)
# plt.title(r"$PXP$ $Z_3$ perts, $\vert \vert [H^z,H^+]-H^+ \vert \vert_F, N=6$")
# plt.show()

# plt.contourf(x,y,M2,levels=np.arange(0,1,0.05))
# plt.colorbar()
# plt.scatter(-0.16439141,0.10456971,marker="x",color="red",s=100)
# plt.xlabel(r"$\lambda_2$")
# plt.ylabel(r"$\lambda_1$",rotation=0)
# plt.title(r"$PXP$ $Z_3$ perts, $\vert \vert [H^z,H^+]-H^+ \vert \vert_F, N=6$")
# plt.show()

# plt.contourf(x,y,M3,levels=np.arange(0,1.05,0.005))
# plt.colorbar()
# plt.scatter(-0.10390399,0.18243653,marker="x",color="red",s=100)
# plt.xlabel(r"$\lambda_2$")
# plt.ylabel(r"$\lambda_1$",rotation=0)
# plt.title(r"$PXP$ $Z_3$ perts, $\vert \vert [H^z,H^+]-H^+ \vert \vert_F, N=6$")
# plt.show()

np.save("pxp,z3_perts,coef_rangex,"+str(pxp.N),x)
np.save("pxp,z3_perts,coef_rangey,"+str(pxp.N),y)
np.save("pxp,z3_perts,M,lie_algebra_error,"+str(pxp.N),M)
np.save("pxp,z3_perts,M,z_var_error,"+str(pxp.N),M2)
np.save("pxp,z3_perts,M,fid_error,"+str(pxp.N),M3)

# coef = np.array([0.18243653,-0.10390499,0.054452])
# # coef = np.array([0.18243653,-0.10390499])
# # coef2 = np.array([0.09982653,-0.17003587,0.01092036])
# # coef = np.array([0.2451307,0,0])
# from scipy.optimize import minimize
# # res = minimize(lambda params: lie_algebra_error(params),method="powell",x0=coef)
# # res = minimize(lambda params: z_var_error(params),method="powell",x0=coef)
# res = minimize(lambda params: fidelity_error(params),method="powell",x0=coef)
# print(res.x)
# lie_algebra_error(res.x,plot=True)
# # res = minimize(lambda params: harmonic_spacing_error(params),method="powell",x0=coef)
# # lie_algebra_error(res.x,plot=True)
# # lie_algebra_error(coef,plot=True)
# # # lie_algebra_error(coef,plot=True)
# # # lie_algebra_error(coef2,plot=True)

