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
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

z=zm_state(3,1,pxp)
e = H.sector.eigvalues()
z3_o = np.log10(np.abs(H.sector.eigvectors()[pxp.keys[z.ref],:])**2)
u = H.sector.eigvectors()

wf = np.load("../z3,entangled_MPS_coef,"+str(pxp.N)+".npy")
psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),wf)
mps_o = np.log10(np.abs(psi_energy)**2)

np.save("pxp,e,"+str(pxp.N),e)
np.save("pxp,u,"+str(pxp.N),u)
np.save("pxp,z3_overlap,"+str(pxp.N),z3_o)
np.save("pxp,mps_overlap,"+str(pxp.N),mps_o)
