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

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=4)])
z=zm_state(4,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

V1_ops = dict()
V1_ops[0] = Hamiltonian(pxp,pxp_syms)
V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V1_ops[0].model = np.array([[0,1,1,1,0]])
V1_ops[0].model_coef = np.array([1])
V1_ops[0].gen(uc_size=4,uc_pos=0)

V1 = V1_ops[0]

V2_ops = dict()
V2_ops[0] = Hamiltonian(pxp,pxp_syms)
V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[0].model = np.array([[0,1,1,1,0]])
V2_ops[0].model_coef = np.array([1])
V2_ops[0].gen(uc_size=4,uc_pos=3)

V2_ops[1] = Hamiltonian(pxp,pxp_syms)
V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V2_ops[1].model = np.array([[0,1,1,1,0]])
V2_ops[1].model_coef = np.array([1])
V2_ops[1].gen(uc_size=4,uc_pos=1)

V2 = V2_ops[0]
for n in range(1,len(V2_ops)):
    V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

V3_ops = dict()
V3_ops[0] = Hamiltonian(pxp,pxp_syms)
V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[0].model = np.array([[0,0,1,0]])
V3_ops[0].model_coef = np.array([1])
V3_ops[0].gen(uc_size=4,uc_pos=0)

V3_ops[1] = Hamiltonian(pxp,pxp_syms)
V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[1].model = np.array([[0,1,0,0]])
V3_ops[1].model_coef = np.array([1])
V3_ops[1].gen(uc_size=4,uc_pos=0)

V3_ops[2] = Hamiltonian(pxp,pxp_syms)
V3_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[2].model = np.array([[0,0,1,0]])
V3_ops[2].model_coef = np.array([1])
V3_ops[2].gen(uc_size=4,uc_pos=1)

V3_ops[3] = Hamiltonian(pxp,pxp_syms)
V3_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V3_ops[3].model = np.array([[0,1,0,0]])
V3_ops[3].model_coef = np.array([1])
V3_ops[3].gen(uc_size=4,uc_pos=1)

V3 = V3_ops[0]
for n in range(1,len(V3_ops)):
    V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

V4_ops = dict()
V4_ops[0] = Hamiltonian(pxp,pxp_syms)
V4_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
V4_ops[0].model = np.array([[0,1,0,0]])
V4_ops[0].model_coef = np.array([1])
V4_ops[0].gen(uc_size=4,uc_pos=2)

V4_ops[1] = Hamiltonian(pxp,pxp_syms)
V4_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
V4_ops[1].model = np.array([[0,0,1,0]])
V4_ops[1].model_coef = np.array([1])
V4_ops[1].gen(uc_size=4,uc_pos=3)

V4_ops[2] = Hamiltonian(pxp,pxp_syms)
V4_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
V4_ops[2].model = np.array([[0,1,0,0]])
V4_ops[2].model_coef = np.array([1])
V4_ops[2].gen(uc_size=4,uc_pos=3)

V4_ops[3] = Hamiltonian(pxp,pxp_syms)
V4_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
V4_ops[3].model = np.array([[0,0,1,0]])
V4_ops[3].model_coef = np.array([1])
V4_ops[3].gen(uc_size=4,uc_pos=2)

V4 = V4_ops[0]
for n in range(1,len(V4_ops)):
    V4=H_operations.add(V4,V4_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f= np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

def fidelity_error(coef,plot=False):
    H = H_operations.add(H0,V1,np.array([1,coef[0]]))
    H = H_operations.add(H,V2,np.array([1,coef[1]]))
    H = H_operations.add(H,V3,np.array([1,coef[2]]))
    H = H_operations.add(H,V3,np.array([1,coef[3]]))
    H.sector.find_eig()
    z=zm_state(4,1,pxp)

    if plot == True:
        fidelity(z,H).plot(np.arange(0,20,0.01),z)
        plt.title(r"$PXP+\lambda_i V_i$, $Z_4$ pert optimization, $N=12$")
        plt.show()

    psi_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t:fidelity_eval(psi_energy,H.sector.eigvalues(),t),method="golden",bracket=(3.0,4.0))
    f_max = fidelity_eval(psi_energy,H.sector.eigvalues(),res.x)
    print(coef,res.x,f_max)
    if res.x<1e-5:
        return 1000
    else:
        return f_max

# d=0.01
# coef_range = np.arange(-0.2,0.2+d,d)
# M = np.zeros((np.size(coef_range),np.size(coef_range)))
# M2 = np.zeros((np.size(coef_range),np.size(coef_range)))
# M3 = np.zeros((np.size(coef_range),np.size(coef_range)))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(coef_range,axis=0))):
    # for m in range(0,np.size(coef_range,axis=0)):
        # M[n,m] = lie_algebra_error(np.array([coef_range[n],coef_range[m],0]))
        # M2[n,m] = z_var_error(np.array([coef_range[n],coef_range[m],0]))
        # M3[n,m] = -fidelity_error(np.array([coef_range[n],coef_range[m],0]))
# x,y = np.meshgrid(coef_range,coef_range)

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

# np.save("pxp,z3_perts,coef_rangex,"+str(pxp.N),x)
# np.save("pxp,z3_perts,coef_rangey,"+str(pxp.N),y)
# np.save("pxp,z3_perts,M,lie_algebra_error,"+str(pxp.N),M)
# np.save("pxp,z3_perts,M,z_var_error,"+str(pxp.N),M2)
# np.save("pxp,z3_perts,M,fid_error,"+str(pxp.N),M3)

# coef = np.array([0,0,0,0])
# # coef = np.array([0.18243653,-0.10390499])
# coef = np.array([0.09982653,-0.17003587,0.01092036,0])
# coef = np.array([-2.129736,1.34358,4.5139e-2,5.27199e-4])
# # # coef = np.array([0.2451307,0,0])
# from scipy.optimize import minimize
# # res = minimize(lambda params: lie_algebra_error(params),method="powell",x0=coef)
# # # res = minimize(lambda params: z_var_error(params),method="powell",x0=coef)
# # res = minimize(lambda params: fidelity_error(params,plot=True),method="powell",x0=coef)
# res = minimize(lambda params: fidelity_error(params),method="powell",x0=coef)
# print(res.x)
# fidelity_error(res.x,plot=True)
fidelity_error(np.array([-1.82121586,1.079754941,-0.00647866,-0.03425407]),plot=True)
# fidelity_error(np.array([-2.129736,1.34358,4.5139e-2,5.27199e-4]),plot=True)
# # res = minimize(lambda params: harmonic_spacing_error(params),method="powell",x0=coef)
# # lie_algebra_error(res.x,plot=True)
# # lie_algebra_error(coef,plot=True)
# # # lie_algebra_error(coef,plot=True)
# # # lie_algebra_error(coef2,plot=True)

