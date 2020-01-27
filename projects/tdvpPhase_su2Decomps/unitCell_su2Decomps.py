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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,inversion
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
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

def repeat_uc(uc,N):
    length0 = np.size(uc)
    uc0 = np.copy(uc)
    if np.abs(N % length0)<1e-5:
        while np.size(uc) != N:
            uc = np.append(uc,uc0)
        return uc
    else:
        print("N not compatible with unit cell length")

def HpFromKey(key):
    Hp = Hamiltonian(pxp)
    Hp.site_ops[1] = np.array([[0,0],[1,0]])
    Hp.site_ops[2] = np.array([[0,1],[0,0]])
    model = []
    for m in range(0,np.size(key,axis=0)):
        if key[m] == 1:
            model.append([0,1,0])
        else:
            model.append([0,2,0])
    Hp.model = model
    Hp.model_coef = np.ones(pxp.N)
    Hp.uc_size = pxp.N*np.ones(pxp.N)
    Hp.uc_pos = np.arange(0,pxp.N)
    Hp.gen()
    return Hp

class su2LW_data:
    def __init__(self,raisingKey,lw,casimir_Eig,Hz_Eig):
        self.raisingKey = raisingKey
        self.lw = lw
        self.casimir_Eig = casimir_Eig
        self.Hz_Eig = Hz_Eig

#init system
N=18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

uc_size = 2
decompBasis = unlocking_System([0,1],"periodic",2,uc_size)
decompBasis.gen_basis()
decompSym = model_sym_data(decompBasis,[inversion(decompBasis)])
block_refs = decompSym.find_block_refs([0])
block_refs = np.delete(block_refs,0,axis=0)
block_basis = np.zeros((np.size(block_refs),uc_size))
for n in range(0,np.size(block_refs,axis=0)):
    block_basis[n] = decompBasis.basis[decompBasis.keys[block_refs[n]]]

from Diagnostics import print_wf
su2RootData = dict()
count = 0
pbar=ProgressBar()
for n in pbar(range(0,np.size(block_basis,axis=0))):
    print(n,np.size(block_basis))
    raisingKey = repeat_uc(block_basis[n],N)
    dictKey = bin_to_int_base_m(raisingKey,pxp.base)
    Hp = HpFromKey(raisingKey)
    Hm = Hp.herm_conj()

    Hx = 1/2 * (Hp.sector.matrix()+Hm.sector.matrix())
    Hy = 1/2j * (Hp.sector.matrix()-Hm.sector.matrix())
    Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())
    C = np.dot(Hx,Hx)+np.dot(Hy,Hy)+np.dot(Hz,Hz)

    ez,uz = np.linalg.eigh(Hz)
    pbar=ProgressBar()
    for m in pbar(range(0,np.size(ez,axis=0))):
        if np.abs(var(C,uz[:,m]))<1e-5:
            if dictKey in list(su2RootData.keys()):
                su2RootData[dictKey][len(su2RootData[dictKey])] = su2LW_data(raisingKey,uz[:,m],exp(C,uz[:,m]),ez[m])
            else:
                su2RootData[dictKey] = dict()
                su2RootData[dictKey][0] = su2LW_data(raisingKey,uz[:,m],exp(C,uz[:,m]),ez[m])

save_obj(su2RootData,"pxp,su2RootData,uc"+str(uc_size)+","+str(N))
# temp = load_obj("./pxp,su2RootData,uc"+str(uc_size)+","+str(N))

