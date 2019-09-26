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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
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

Hp_ops = dict()
Hp_ops[0] = Hamiltonian(pxp)
Hp_ops[0].site_ops[1] = np.array([[0,1],[0,0]])
Hp_ops[0].model = np.array([[0,1,0]])
Hp_ops[0].model_coef = np.array([1])
Hp_ops[0].gen(uc_size=3,uc_pos=0)

Hp_ops[1] = Hamiltonian(pxp)
Hp_ops[1].site_ops[1] = np.array([[0,1],[0,0]])
Hp_ops[1].model = np.array([[0,1,0]])
Hp_ops[1].model_coef = np.array([1])
Hp_ops[1].gen(uc_size=3,uc_pos=1)

Hp_ops[2] = Hamiltonian(pxp)
Hp_ops[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp_ops[2].model = np.array([[0,1,0]])
Hp_ops[2].model_coef = np.array([1])
Hp_ops[2].gen(uc_size=3,uc_pos=2)

Hp = Hp_ops[0].sector.matrix()
for n in range(1,len(Hp_ops)):
    Hp = Hp + Hp_ops[n].sector.matrix()
Hm = np.conj(np.transpose(Hp))
temp = Hp + Hm
H=spin_Hamiltonian(pxp,"x")
H.gen()
print((np.abs(H.sector.matrix()-temp)<1e-5).all())
    
z=zm_state(3,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,int(2*pxp.N/3)):
    next_state = np.dot(Hm,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()

H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e,u = np.linalg.eigh(H_fsa)
overlap = np.log10(np.abs(u[0,:])**2)
plt.scatter(e,overlap,marker="x",s=100,color="red",label="FSA")
eig_overlap(z,H).plot()
plt.legend()
plt.show()
    
