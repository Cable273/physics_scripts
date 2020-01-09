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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,2,0],[0,1,0],[0,1,0],[0,1,0]])
Hp[0].model_coef = np.array([1,1,1,1])
Hp[0].uc_size = np.array([4,4,4,4])
Hp[0].uc_pos = np.array([3,0,1,2])

# Hp[1] = Hamiltonian(pxp)
# Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[1].model = np.array([[0,0,1,0],[0,1,0,0],[0,2,0,0],[0,0,2,0]])
# Hp[1].model_coef = np.array([1,1,1,1])
# Hp[1].uc_size = np.array([4,4,4,4])
# Hp[1].uc_pos = np.array([3,2,3,2])

# Hp[2] = Hamiltonian(pxp)
# Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[2].model = np.array([[0,0,1,0,0],[0,0,1,0,0]])
# Hp[2].model_coef = np.array([1,1])
# Hp[2].uc_size = np.array([4,4])
# Hp[2].uc_pos = np.array([3,1])

# Hp[3] = Hamiltonian(pxp)
# Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[3].model = np.array([[0,0,0,1,0],[0,1,0,0,0]])
# Hp[3].model_coef = np.array([1,1])
# Hp[3].uc_size = np.array([4,4])
# Hp[3].uc_pos = np.array([3,1])

# Hp[4] = Hamiltonian(pxp)
# Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[4].model = np.array([[0,0,4,0,1,0],[0,1,0,4,0,0]])
# Hp[4].model_coef = np.array([1,1])
# Hp[4].uc_size = np.array([4,4])
# Hp[4].uc_pos = np.array([3,0])

# Hp[5] = Hamiltonian(pxp)
# Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[5].model = np.array([[0,2,0,4,0],[0,4,0,2,0]])
# Hp[5].model_coef = np.array([1,1])
# Hp[5].uc_size = np.array([4,4])
# Hp[5].uc_pos = np.array([3,1])

# Hp[5] = Hamiltonian(pxp)
# Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[5].model = np.array([[0,2,0,3,0],[0,3,0,2,0]])
# Hp[5].model_coef = np.array([1,1])
# Hp[5].uc_size = np.array([4,4])
# Hp[5].uc_pos = np.array([3,1])

# Hp[6] = Hamiltonian(pxp)
# Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[6].model = np.array([[0,2,0,3,0,0],[0,0,3,0,2,0]])
# Hp[6].model_coef = np.array([1,1])
# Hp[6].uc_size = np.array([4,4])
# Hp[6].uc_pos = np.array([3,0])

# Hp[6] = Hamiltonian(pxp)
# Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[6].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
# Hp[6].model_coef = np.array([1,1])
# Hp[6].uc_size = np.array([4,4])
# Hp[6].uc_pos = np.array([3,1])

# Hp[7] = Hamiltonian(pxp)
# Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[7].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[7].model = np.array([[0,1,0,0,0],[0,0,0,1,0]])
# Hp[7].model_coef = np.array([1,1])
# Hp[7].uc_size = np.array([4,4])
# Hp[7].uc_pos = np.array([2,2])

# Hp[8] = Hamiltonian(pxp)
# Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[8].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[8].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0]])
# Hp[8].model_coef = np.array([1,1])
# Hp[8].uc_size = np.array([4,4])
# Hp[8].uc_pos = np.array([1,2])

# Hp[9] = Hamiltonian(pxp)
# Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[9].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[9].model = np.array([[0,3,0,1,0,0],[0,0,1,0,3,0]])
# Hp[9].model_coef = np.array([1,1])
# Hp[9].uc_size = np.array([4,4])
# Hp[9].uc_pos = np.array([3,0])

# Hp[10] = Hamiltonian(pxp)
# Hp[10].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[10].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[10].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[10].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[10].model = np.array([[0,0,2,0,0]])
# Hp[10].model_coef = np.array([1])
# Hp[10].uc_size = np.array([4])
# Hp[10].uc_pos = np.array([2])

# Hp[11] = Hamiltonian(pxp)
# Hp[11].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[11].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[11].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[11].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[11].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
# Hp[11].model_coef = np.array([1,1,1,1])
# Hp[11].uc_size = np.array([4,4,4,4])
# Hp[11].uc_pos = np.array([0,0,1,1])

# Hp[12] = Hamiltonian(pxp)
# Hp[12].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[12].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[12].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[12].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[12].model = np.array([[0,0,1,0,0]])
# Hp[12].model_coef = np.array([1])
# Hp[12].uc_size = np.array([4])
# Hp[12].uc_pos = np.array([0])

# Hp[13] = Hamiltonian(pxp)
# Hp[13].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[13].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[13].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[13].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[13].model = np.array([[0,1,0,0,0],[0,0,0,1,0]])
# Hp[13].model_coef = np.array([1,1])
# Hp[13].uc_size = np.array([4,4])
# Hp[13].uc_pos = np.array([0,0])

# Hp[14] = Hamiltonian(pxp)
# Hp[14].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[14].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[14].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[14].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[14].model = np.array([[0,1,0,4,0],[0,4,0,1,0]])
# Hp[14].model_coef = np.array([1,1])
# Hp[14].uc_size = np.array([4,4])
# Hp[14].uc_pos = np.array([0,0])

# Hp[15] = Hamiltonian(pxp)
# Hp[15].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[15].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[15].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[15].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[15].model = np.array([[0,0,4,0,2,0],[0,2,0,4,0,0]])
# Hp[15].model_coef = np.array([1,1])
# Hp[15].uc_size = np.array([4,4])
# Hp[15].uc_pos = np.array([0,3])

# Hp[16] = Hamiltonian(pxp)
# Hp[16].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[16].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[16].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[16].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[16].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0]])
# Hp[16].model_coef = np.array([1,1])
# Hp[16].uc_size = np.array([4,4])
# Hp[16].uc_pos = np.array([0,3])

# Hp[17] = Hamiltonian(pxp)
# Hp[17].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[17].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[17].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[17].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[17].model = np.array([[0,4,0,1,0],[0,1,0,4,0]])
# Hp[17].model_coef = np.array([1,1])
# Hp[17].uc_size = np.array([4,4])
# Hp[17].uc_pos = np.array([3,1])

# Hp[18] = Hamiltonian(pxp)
# Hp[18].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[18].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[18].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[18].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[18].model = np.array([[0,0,3,0,1,0],[0,1,0,3,0,0]])
# Hp[18].model_coef = np.array([1,1])
# Hp[18].uc_size = np.array([4,4])
# Hp[18].uc_pos = np.array([2,1])

# Hp[19] = Hamiltonian(pxp)
# Hp[19].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[19].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[19].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[19].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[19].model = np.array([[0,2,0,0,0],[0,0,0,2,0]])
# Hp[19].model_coef = np.array([1,1])
# Hp[19].uc_size = np.array([4,4])
# Hp[19].uc_pos = np.array([3,1])

# Hp[20] = Hamiltonian(pxp)
# Hp[20].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[20].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[20].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[20].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[20].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
# Hp[20].model_coef = np.array([1,1])
# Hp[20].uc_size = np.array([4,4])
# Hp[20].uc_pos = np.array([0,0])

# Hp[21] = Hamiltonian(pxp)
# Hp[21].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[21].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[21].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[21].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[21].model = np.array([[0,3,0,1,0,0],[0,0,1,0,3,0]])
# Hp[21].model_coef = np.array([1,1])
# Hp[21].uc_size = np.array([4,4])
# Hp[21].uc_pos = np.array([0,3])

# Hp[22] = Hamiltonian(pxp)
# Hp[22].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[22].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[22].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[22].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[22].model = np.array([[0,0,2,0,0,0],[0,0,0,2,0,0]])
# Hp[22].model_coef = np.array([1,1])
# Hp[22].uc_size = np.array([4,4])
# Hp[22].uc_pos = np.array([2,1])

# Hp[23] = Hamiltonian(pxp)
# Hp[23].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[23].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[23].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[23].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[23].model = np.array([[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,3,0,1,0],[0,1,0,3,0,0]])
# Hp[23].model_coef = np.array([1,1,1,1])
# Hp[23].uc_size = np.array([4,4,4,4])
# Hp[23].uc_pos = np.array([0,1,3,2])

# Hp[24] = Hamiltonian(pxp)
# Hp[24].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[24].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[24].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[24].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[24].model = np.array([[0,3,0,1,0],[0,1,0,3,0]])
# Hp[24].model_coef = np.array([1,1])
# Hp[24].uc_size = np.array([4,4])
# Hp[24].uc_pos = np.array([2,2])

# Hp[25] = Hamiltonian(pxp)
# Hp[25].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[25].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[25].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[25].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[25].model = np.array([[0,0,1,0,3,0],[0,3,0,1,0,0]])
# Hp[25].model_coef = np.array([1,1])
# Hp[25].uc_size = np.array([4,4])
# Hp[25].uc_pos = np.array([1,2])

# Hp[26] = Hamiltonian(pxp)
# Hp[26].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[26].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[26].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[26].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[26].model = np.array([[0,0,0,1,0,0],[0,0,1,0,0,0]])
# Hp[26].model_coef = np.array([1,1])
# Hp[26].uc_size = np.array([4,4])
# Hp[26].uc_pos = np.array([0,3])

# Hp[27] = Hamiltonian(pxp)
# Hp[27].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[27].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[27].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[27].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[27].model = np.array([[0,0,2,0,3,0],[0,3,0,2,0,0]])
# Hp[27].model_coef = np.array([1,1])
# Hp[27].uc_size = np.array([4,4])
# Hp[27].uc_pos = np.array([2,1])

# Hp[28] = Hamiltonian(pxp)
# Hp[28].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[28].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[28].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[28].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[28].model = np.array([[0,0,4,0,2,0,0],[0,0,2,0,4,0,0],[0,1,0,4,0,4,0],[0,4,0,4,0,1,0]])
# Hp[28].model_coef = np.array([1,1,1,1])
# Hp[28].uc_size = np.array([4,4,4,4])
# Hp[28].uc_pos = np.array([0,2,2,0])

# Hp[29] = Hamiltonian(pxp)
# Hp[29].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[29].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[29].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[29].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[29].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
# Hp[29].model_coef = np.array([1,1])
# Hp[29].uc_size = np.array([4,4])
# Hp[29].uc_pos = np.array([0,3])

# Hp[30] = Hamiltonian(pxp)
# Hp[30].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[30].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[30].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[30].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[30].model = np.array([[0,0,3,0,2,0,0],[0,0,2,0,3,0,0]])
# Hp[30].model_coef = np.array([1,1])
# Hp[30].uc_size = np.array([4,4])
# Hp[30].uc_pos = np.array([0,2])

# Hp[31] = Hamiltonian(pxp)
# Hp[31].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[31].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[31].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[31].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[31].model = np.array([[0,0,3,0,1,0,0],[0,0,1,0,3,0,0]])
# Hp[31].model_coef = np.array([1,1])
# Hp[31].uc_size = np.array([4,4])
# Hp[31].uc_pos = np.array([1,1])

# Hp[32] = Hamiltonian(pxp)
# Hp[32].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[32].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[32].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[32].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[32].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
# Hp[32].model_coef = np.array([1,1])
# Hp[32].uc_size = np.array([4,4])
# Hp[32].uc_pos = np.array([2,1])

# Hp[33] = Hamiltonian(pxp)
# Hp[33].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[33].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[33].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[33].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[33].model = np.array([[0,0,3,0,1,0,0],[0,0,1,0,3,0,0]])
# Hp[33].model_coef = np.array([1,1])
# Hp[33].uc_size = np.array([4,4])
# Hp[33].uc_pos = np.array([3,3])

# Hp[34] = Hamiltonian(pxp)
# Hp[34].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[34].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[34].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[34].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[34].model = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]])
# Hp[34].model_coef = np.array([1,1])
# Hp[34].uc_size = np.array([4,4])
# Hp[34].uc_pos = np.array([1,2])

# Hp[35] = Hamiltonian(pxp)
# Hp[35].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[35].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[35].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[35].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[35].model = np.array([[0,0,1,0,3,0,0],[0,0,3,0,1,0,0]])
# Hp[35].model_coef = np.array([1,1])
# Hp[35].uc_size = np.array([4,4])
# Hp[35].uc_pos = np.array([0,2])

# Hp[36] = Hamiltonian(pxp)
# Hp[36].site_ops[1] = np.array([[0,0],[1,0]])
# Hp[36].site_ops[2] = np.array([[0,1],[0,0]])
# Hp[36].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
# Hp[36].site_ops[4] = np.array([[0,0],[0,1]])
# Hp[36].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
# Hp[36].model_coef = np.array([1,1])
# Hp[36].uc_size = np.array([4,4])
# Hp[36].uc_pos = np.array([3,1])

Hp[1] = Hamiltonian(pxp)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
Hp[1].model = np.array([[0,2,1,2,0],[0,2,1,2,0]])
Hp[1].model_coef = np.array([1,1])
Hp[1].uc_size = np.array([4,4])
Hp[1].uc_pos = np.array([3,1])

Hp[2] = Hamiltonian(pxp)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[0,1,2,1,0]])
Hp[2].model_coef = np.array([1])
Hp[2].uc_size = np.array([4])
Hp[2].uc_pos = np.array([0])

Hp[3] = Hamiltonian(pxp)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].site_ops[4] = np.array([[0,0],[0,1]])
Hp[3].model = np.array([[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
Hp[3].model_coef = np.array([1,1,1,1])
Hp[3].uc_size = np.array([4,4,4,4])
Hp[3].uc_pos = np.array([0,0,1,1])

Hp[4] = Hamiltonian(pxp)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
Hp[4].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Hp[4].model_coef = np.array([1,1,1,1])
Hp[4].uc_size = np.array([4,4,4,4])
Hp[4].uc_pos = np.array([2,3,3,2])


# Hp[0].gen()
for n in range(0,len(Hp)):
    Hp[n].gen()

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

from copy import deepcopy
from Hamiltonian_Classes import H_operations
from scipy.optimize import minimize,minimize_scalar
def fidelity_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    e,u = np.linalg.eigh(H)
    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

    t=np.arange(0,10,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        f[n] = -fidelity_eval(psi_energy,e,t[n])
    # plt.plot(t,f)
    # plt.show()
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f_max = np.max(f[cut:])
        
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(4.5,5.5))
    f = -fidelity_eval(psi_energy,e,res.x)
    print(coef,f)
    # print(f)
    if res.x <1e-5:
        return 1000
    else:
        return -f_max
    # if (np.abs(coef)>0.5).any():
        # return 1000
    # return -f

from Calculations import get_top_band_indices
def spacing_error(coef):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    e,u = np.linalg.eigh(H)
    z=zm_state(4,1,pxp)
    psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())
    overlap = np.log10(np.abs(psi_energy)**2)
    scar_indices = get_top_band_indices(e,overlap,int(2*N/3),150,200,e_diff=0.5)
    # plt.scatter(e,overlap)
    # for n in range(0,np.size(scar_indices,axis=0)):
        # plt.scatter(e[scar_indices[n]],overlap[scar_indices[n]],marker="x",s=100,color="red")
    # plt.show()

    scar_e = np.zeros(np.size(scar_indices))
    for n in range(0,np.size(scar_indices,axis=0)):
        scar_e[n] = e[scar_indices[n]]
    diffs = np.zeros(np.size(scar_e)-1)
    for n in range(0,np.size(diffs,axis=0)):
        diffs[n] = scar_e[n+1] - scar_e[n]
    diff_matrix = np.zeros((np.size(diffs),np.size(diffs)))
    for n in range(0,np.size(diff_matrix,axis=0)):
        for m in range(0,np.size(diff_matrix,axis=0)):
            diff_matrix[n,m] = np.abs(diffs[n] - diffs[m])
    error = np.power(np.trace(np.dot(diff_matrix,np.conj(np.transpose(diff_matrix)))),0.5)
    print(coef,error)
    print(scar_e)
    # print(coef)
    # if (np.abs(coef).any())>0.5:
        # return 1000
    # else:
    return error

# coef = np.ones(4)*0.05
coef = np.zeros(4)
coef[0] = -1
# coef = np.array([-1.43,0.0008,0.0979,0.0980])
# # coef = np.array([0.00152401,-1.42847732,0.09801341,0.09784458])
# coef = np.load("./pxp,z4,2nd_order_pert_coef,12.npy")
# coef = np.zeros(np.size(coef))
# coef[np.size(coef)-1] = -1.43
# res = minimize(lambda coef:fidelity_error(coef),method="nelder-mead",x0=coef,tol=1e-10)
# coef = res.x
# res = minimize(lambda coef:spacing_error(coef),method="powell",x0=coef)
# coef = res.x
# coef = np.zeros(4)

print(coef)
error = spacing_error(coef)
print("ERROR VALUE: "+str(error))

print(coef)
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hp = Hp_total.sector.matrix()
Hm = np.conj(np.transpose(Hp))

H = Hp+Hm
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 1/2*com(Hp,Hm)

e,u = np.linalg.eigh(H)
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
z=zm_state(4,1,pxp)
psi_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(3.5,5.5))
f0 = fidelity_eval(psi_energy,e,res.x)
print("F0 REVIVAL")
print(res.x,fidelity_eval(psi_energy,e,res.x))
# np.savetxt("pxp,z4,f0,"+str(pxp.N),[f0])

z2=zm_state(4,1,pxp,1)
z3=zm_state(4,1,pxp,2)
z4=zm_state(4,1,pxp,3)
for n in range(0,np.size(t,axis=0)):
    f[n] = -fidelity_eval(psi_energy,e,t[n])
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$PXP, Z_4$, $-V_1$ SU(2) pertubation, N="+str(pxp.N))
plt.show()

overlap = np.log10(np.abs(psi_energy)**2)
eigenvalues = np.copy(e)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
    
plt.scatter(eigenvalues,overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_4 \vert E \rangle \vert^2)$")
plt.title(r"$PXP, Z_4$, $-V_1$ SU(2) pertubation, N="+str(pxp.N))
plt.show()

#check harmonic spacing
def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

z=zm_state(4,1,pxp)

fsa_basis = z.prod_basis()
current_state = fsa_basis
fsa_dim = int(2*(pxp.N/4))
for n in range(0,fsa_dim):
    next_state = norm(np.dot(Hp,current_state))
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)
    
Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
for n in range(0,np.size(fsa_basis,axis=1)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])
    print(Hz_exp[n],Hz_var[n])

print("\n")
e_diff = np.zeros(np.size(Hz_exp)-1)
for n in range(0,np.size(e_diff,axis=0)):
    e_diff[n] = Hz_exp[n+1]-Hz_exp[n]
    print(e_diff[n])

# np.save("z4,Hz_diff,16",e_diff)
    
np.save("pxp,z4,1st_order_pert_coef,"+str(pxp.N),coef)
np.save("z4,Hz_variance,"+str(pxp.N),Hz_var)
np.savetxt("pxp,z4,Hz_eigvalues,"+str(pxp.N),Hz_exp)
np.savetxt("pxp,z4,Hz_var,"+str(pxp.N),Hz_var)
np.savetxt("pxp,z4,Hz_eig_diffs,"+str(pxp.N),e_diff)
np.savetxt("pxp,z4,f0,"+str(pxp.N),[f0])
np.savetxt("pxp,z4,spacing_error,"+str(pxp.N),[error])
