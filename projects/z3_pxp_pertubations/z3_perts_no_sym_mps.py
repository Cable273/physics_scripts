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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
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

N = 18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])
z=zm_state(3,1,pxp)

# V1_ops = dict()
# V1_ops[0] = Hamiltonian(pxp,pxp_syms)
# V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
# V1_ops[0].model = np.array([[0,1,0,0]])
# V1_ops[0].model_coef = np.array([1])
# V1_ops[0].gen(uc_size=3,uc_pos=1)

# V1_ops[1] = Hamiltonian(pxp,pxp_syms)
# V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
# V1_ops[1].model = np.array([[0,0,1,0]])
# V1_ops[1].model_coef = np.array([1])
# V1_ops[1].gen(uc_size=3,uc_pos=2)

# V1_ops[2] = Hamiltonian(pxp,pxp_syms)
# V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
# V1_ops[2].model = np.array([[0,1,0,0]])
# V1_ops[2].model_coef = np.array([1])
# V1_ops[2].gen(uc_size=3,uc_pos=2)

# V1_ops[3] = Hamiltonian(pxp,pxp_syms)
# V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
# V1_ops[3].model = np.array([[0,0,1,0]])
# V1_ops[3].model_coef = np.array([1])
# V1_ops[3].gen(uc_size=3,uc_pos=1)

# V1 = V1_ops[0]
# for n in range(1,len(V1_ops)):
    # V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

# V2_ops = dict()
# V2_ops[0] = Hamiltonian(pxp,pxp_syms)
# V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
# V2_ops[0].model = np.array([[0,0,1,0]])
# V2_ops[0].model_coef = np.array([1])
# V2_ops[0].gen(uc_size=3,uc_pos=0)

# V2_ops[1] = Hamiltonian(pxp,pxp_syms)
# V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
# V2_ops[1].model = np.array([[0,1,0,0]])
# V2_ops[1].model_coef = np.array([1])
# V2_ops[1].gen(uc_size=3,uc_pos=0)

# V2 = V2_ops[0]
# for n in range(1,len(V2_ops)):
    # V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

# V3_ops = dict()
# V3_ops[0] = Hamiltonian(pxp,pxp_syms)
# V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
# V3_ops[0].model = np.array([[0,1,1,1,0]])
# V3_ops[0].model_coef = np.array([1])
# V3_ops[0].gen(uc_size=3,uc_pos=0)

# V3_ops[1] = Hamiltonian(pxp,pxp_syms)
# V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
# V3_ops[1].model = np.array([[0,1,1,1,0]])
# V3_ops[1].model_coef = np.array([1])
# V3_ops[1].gen(uc_size=3,uc_pos=2)

# V3 = V3_ops[0]
# for n in range(1,len(V3_ops)):
    # V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()
# coef = np.array([0.18243653,-0.10390499,0.054452])
# coef = np.array([0.00783215,0.16966656,-0.05180238])
# H = H_operations.add(H0,V1,np.array([1,coef[0]]))
# H = H_operations.add(H,V2,np.array([1,coef[1]]))
# H = H_operations.add(H,V3,np.array([1,coef[2]]))
H = H0

H.sector.find_eig()
psi = np.load("./z3,entangled_MPS_coef,"+str(pxp.N)+".npy")
psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),psi)
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(),t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
np.save("pxp,z3_mps,no_pert,t,"+str(pxp.N),t)
np.save("pxp,z3_mps,no_pert,fidelity,"+str(pxp.N),f)
plt.plot(t,f)
plt.show()
