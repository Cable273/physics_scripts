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
from Symmetry_Classes import *
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':36})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 9
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational_general(pxp,order=3),PT(pxp)])
    
z=zm_state(3,1,pxp)
k=pxp_syms.find_k_ref(z.ref)

V1 = Hamiltonian(pxp,pxp_syms)
V1.site_ops[1] = np.array([[0,1],[1,0]])
V1.model = np.array([[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0]])
V1.model_coef = np.array([1,1,1,1])
V1.uc_size = np.array([3,3,3,3])
V1.uc_pos = np.array([1,2,2,1])
for n in range(0,np.size(k,axis=0)):
    V1.gen(k_vec=k[n])

V2 = Hamiltonian(pxp,pxp_syms)
V2.site_ops[1] = np.array([[0,1],[1,0]])
V2.model = np.array([[0,0,1,0],[0,1,0,0]])
V2.model_coef = np.array([1,1])
V2.uc_size = np.array([3,3])
V2.uc_pos = np.array([0,0])
for n in range(0,np.size(k,axis=0)):
    V2.gen(k_vec=k[n])

V3 = Hamiltonian(pxp,pxp_syms)
V3.site_ops[1] = np.array([[0,1],[1,0]])
V3.model = np.array([[0,1,1,1,0],[0,1,1,1,0]])
V3.model_coef = np.array([1,1])
V3.uc_size = np.array([3,3])
V3.uc_pos = np.array([0,2])
for n in range(0,np.size(k,axis=0)):
    V3.gen(k_vec=k[n])

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
for n in range(0,np.size(k,axis=0)):
    H0.gen(k[n])

V1.gen()
V2.gen()
V3.gen()
H0.gen()

coef = np.array([0.18243653,-0.10390499,0.054452])
H = H_operations.add(H0,V1,np.array([1,coef[0]]))
H = H_operations.add(H,V2,np.array([1,coef[1]]))
H = H_operations.add(H,V3,np.array([1,coef[2]]))

coef = np.array([0.09992653,-0.17003587,0.01092036])
H1 = H_operations.add(H0,V1,np.array([1,coef[0]]))
H1 = H_operations.add(H1,V2,np.array([1,coef[1]]))
H1 = H_operations.add(H1,V3,np.array([1,coef[2]]))

coef = np.array([0.10456971,-0.16439141,0.03094977])
H2 = H_operations.add(H0,V1,np.array([1,coef[0]]))
H2 = H_operations.add(H2,V2,np.array([1,coef[1]]))
H2 = H_operations.add(H2,V3,np.array([1,coef[2]]))

# coef = np.array([0.14642557,-0.16027117,0.0218123])
coef = np.array([0.1494363,-0.16027541,0.02084993])
H3 = H_operations.add(H0,V1,np.array([1,coef[0]]))
H3 = H_operations.add(H3,V2,np.array([1,coef[1]]))
H3 = H_operations.add(H3,V3,np.array([1,coef[2]]))

for n in range(0,np.size(k,axis=0)):
    H.sector.find_eig(k[n])
    H1.sector.find_eig(k[n])
    H2.sector.find_eig(k[n])
    H3.sector.find_eig(k[n])
H.sector.find_eig()
print(H.sector.eigvalues())
print(H.sector.eigvalues(k[0]))

# t=np.arange(0,20,0.01)
# f = fidelity(z,H,"use sym").eval(t,z)
# f1 = fidelity(z,H1,"use sym").eval(t,z)
# f2 = fidelity(z,H2,"use sym").eval(t,z)
# f3 = fidelity(z,H3,"use sym").eval(t,z)
# np.save("pxp,z3_perts,t,"+str(pxp.N),t)
# np.save("pxp,z3_perts,f,f0,"+str(pxp.N),f)
# np.save("pxp,z3_perts,f,lie_algebra,"+str(pxp.N),f1)
# np.save("pxp,z3_perts,f,hz_var,"+str(pxp.N),f2)
# np.save("pxp,z3_perts,f,subspace_var,"+str(pxp.N),f3)

# plt.plot(t,f,linewidth=2,label=r"$f_0$")
# plt.plot(t,f1,linewidth=2,alpha=1.0,label=r"Lie algebra")
# plt.plot(t,f2,linewidth=2,alpha=1.0,label=r"$H^z$ Variance")
# plt.plot(t,f3,linewidth=2,alpha=1.0,label=r"Subspace Variance")
# plt.legend()
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# plt.title(r"$PXP + \lambda_i V+i$, $Z_3$ pertubations, $N=$"+str(pxp.N))
# plt.show()

