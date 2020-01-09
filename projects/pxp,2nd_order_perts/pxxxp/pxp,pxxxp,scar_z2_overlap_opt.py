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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def norm(psi):
    return psi / np.power(np.vdot(psi,psi),0.5)
def exp(Q,psi):
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0 = spin_Hamiltonian(pxp,"x")

V = Hamiltonian(pxp)
V.site_ops[1] = np.array([[0,1],[1,0]])
V.model = np.array([[0,1,1,1,0]])
V.model_coef = np.array([1])

# V.model = np.array([[0,1,0,0],[0,0,1,0]])
# V.model_coef = np.array([1,1])

H0.gen()
V.gen()

z=zm_state(2,1,pxp)
#get scar z2 overlap (no pert)
H0.sector.find_eig()
overlap = eig_overlap(z,H0).eval()
from Calculations import get_top_band_indices
scar_indices = get_top_band_indices(H0.sector.eigvalues(),overlap,pxp.N,200,150,0.8)

#check got right scars
plt.scatter(H0.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H0.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
plt.show()

scar_overlap = np.zeros(np.size(scar_indices))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_overlap[n] = overlap[scar_indices[n]]

#optimize for z2 scar overlap with pert
def overlap_error(coef):
    # coef = coef[0]
    H = H_operations.add(H0,V,np.array([1,coef]))
    H.sector.find_eig()
    overlap_no_log = np.abs(H.sector.eigvectors()[pxp.keys[z.ref],:])**2
    overlap = np.log10(overlap_no_log)
    scar_indices_perturbed = np.flip(get_top_band_indices(H.sector.eigvalues(),overlap,pxp.N,200,150,0.8))

    # check got right scars
    # plt.scatter(H.sector.eigvalues(),overlap)
    # for n in range(0,np.size(scar_indices_perturbed,axis=0)):
        # plt.scatter(H.sector.eigvalues()[scar_indices_perturbed[n]],overlap[scar_indices_perturbed[n]],marker="x",color="red",s=100)
    # plt.show()

    for n in range(0,np.size(scar_indices_perturbed,axis=0)):
        overlap_no_log = np.delete(overlap_no_log,scar_indices_perturbed[n],axis=0)
    cost = np.sum(overlap_no_log)
        

    # cost = np.sum(scar_overlap_perturbed-scar_overlap)
    print(coef,cost)
    return cost

from scipy.optimize import minimize_scalar
# coef = [0.122]
coef = [0]
# coef = [0.13698226353038712]
res = minimize_scalar(lambda coef: overlap_error(coef),method="Golden")
coef = res.x
# coef = coef[0]


H = H_operations.add(H0,V,np.array([1,coef]))
H.sector.find_eig()
eig_overlap(z,H).plot()
plt.show()
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()




