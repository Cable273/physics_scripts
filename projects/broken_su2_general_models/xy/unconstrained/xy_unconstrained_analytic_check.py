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
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

#init system
N=8
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

Hp = Hamiltonian(pxp,pxp_syms)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[1,2],[2,1]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([2,2])
Hp.uc_pos = np.array([0,1])

Hp.gen()
Hm = Hp.herm_conj()
Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())

lie_algebra_num = com(Hz,Hp.sector.matrix())

Hz_an = Hamiltonian(pxp,pxp_syms)
Hz_an.site_ops[1] = np.array([[0,0],[1,0]])
Hz_an.site_ops[2] = np.array([[0,1],[0,0]])
Hz_an.site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hz_an.site_ops[4] = np.array([[0,0],[0,1]])
Hz_an.model = np.array([[0,4],[4,0],[0,4],[4,0],[1,3,2],[2,3,1],[1,3,2],[2,3,1]])
Hz_an.model_coef = np.array([1/2,1/2,-1/2,-1/2,1,1,-1,-1])
Hz_an.uc_size = np.array([2,2,2,2,2,2,2,2])
Hz_an.uc_pos = np.array([1,0,0,1,1,1,0,0])
Hz_an.gen()
print((np.abs(Hz-Hz_an.sector.matrix())<1e-5).all())

lie_an = Hamiltonian(pxp,pxp_syms)
lie_an.site_ops[1] = np.array([[0,0],[1,0]])
lie_an.site_ops[2] = np.array([[0,1],[0,0]])
lie_an.site_ops[3] = np.array([[-1/2,0],[0,1/2]])
lie_an.site_ops[4] = np.array([[0,0],[0,1]])
lie_an.model = np.array([[1,2],[2,1],[0,1,2],[4,1,2],[1,2,0],[1,2,4],[0,2,1],[4,2,1],[2,1,0],[2,1,4],[1,3,3,2],[2,3,3,1]])
lie_an.model_coef = np.array([1,1,1,1,1,1,1,1,1,1,4,4,])
lie_an.uc_size = np.array([2,2,2,2,2,2,2,2,2,2,2,2])
lie_an.uc_pos = np.array([0,1,1,1,0,0,0,0,1,1,0,1])
lie_an.gen()
plt.matshow(np.abs(lie_algebra_num))
plt.matshow(np.abs(lie_an.sector.matrix()))
plt.show()
print((np.abs(lie_algebra_num-lie_an.sector.matrix())<1e-5).all())


