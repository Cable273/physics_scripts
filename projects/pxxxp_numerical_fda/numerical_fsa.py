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

N = 24
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)

V1 = Hamiltonian(pxp,pxp_syms)
V1.site_ops[1] = np.array([[0,1],[1,0]])
V1.site_ops[2] = np.array([[-1,0],[0,1]])
V1.model = np.array([[0,0,1,0],[1,0,0,0]])
V1.model_coef = np.array([1,1])

V2 = Hamiltonian(pxp,pxp_syms)
V2.site_ops[1] = np.array([[0,1],[1,0]])
V2.site_ops[2] = np.array([[-1,0],[0,1]])
V2.model = np.array([[0-1,0,1,0],[0,1,0,-1,0]])
V2.model_coef = np.array([1,1])

V3 = Hamiltonian(pxp,pxp_syms)
V3.site_ops[1] = np.array([[0,1],[1,0]])
V3.site_ops[2] = np.array([[-1,0],[0,1]])
V3.model = np.array([[0-1,-1,0,1,0],[0,1,0,-1,-1,0]])
V3.model_coef = np.array([1,1])

V4 = Hamiltonian(pxp,pxp_syms)
V4.site_ops[1] = np.array([[0,1],[1,0]])
V4.site_ops[2] = np.array([[-1,0],[0,1]])
V4.model = np.array([[0-1,-1,-1,0,1,0],[0,1,0,-1,-1,-1,0]])
V4.model_coef = np.array([1,1])

# H0.gen()
# V1.gen()
# V2.gen()
# V3.gen()
# V4.gen()
z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H0.gen(k[n])
    V1.gen(k[n])
    V2.gen(k[n])
    V3.gen(k[n])
    V4.gen(k[n])
    

from Hamiltonian_Classes import H_operations
def h(n):
    phi = (1+np.power(5,0.5))/2
    # h0 = 0.051
    h0 = 0.108
    val = h0/np.power(np.power(phi,n-1)-np.power(phi,-(n-1)),2)
    return val
d=np.arange(2,6)
c=np.zeros(np.size(d))
for n in range(0,np.size(c,axis=0)):
    c[n] = h(d[n])

H = H_operations.add(H0,V1,np.array([1,c[0]]))
H = H_operations.add(H,V2,np.array([1,c[1]]))
H = H_operations.add(H,V3,np.array([1,c[2]]))
H = H_operations.add(H,V4,np.array([1,c[3]]))

for n in range(0,np.size(k,axis=0)):
    H.sector.find_eig(k[n])
    eig_overlap(z,H,k[n]).plot()
plt.show()
fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
plt.show()
