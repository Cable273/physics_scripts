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
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N_vals = np.arange(12,24,3)
for count in range(0,np.size(N_vals,axis=0)):
    N = N_vals[count]
    pxp = unlocking_System([0],"periodic",2,N)
    pxp.gen_basis()
    pxp_syms=model_sym_data(pxp,[translational_general(pxp,order=3),PT(pxp)])
    z=zm_state(3,1,pxp)
    k=pxp_syms.find_k_ref(z.ref)

    V1_ops = dict()
    V1_ops[0] = Hamiltonian(pxp,pxp_syms)
    V1_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
    V1_ops[0].model = np.array([[0,1,0,0]])
    V1_ops[0].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V1_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=1)

    V1_ops[1] = Hamiltonian(pxp,pxp_syms)
    V1_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
    V1_ops[1].model = np.array([[0,0,1,0]])
    V1_ops[1].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V1_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

    V1_ops[2] = Hamiltonian(pxp,pxp_syms)
    V1_ops[2].site_ops[1] = np.array([[0,1],[1,0]])
    V1_ops[2].model = np.array([[0,1,0,0]])
    V1_ops[2].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V1_ops[2].gen(k_vec=k[n],uc_size=3,uc_pos=2)

    V1_ops[3] = Hamiltonian(pxp,pxp_syms)
    V1_ops[3].site_ops[1] = np.array([[0,1],[1,0]])
    V1_ops[3].model = np.array([[0,0,1,0]])
    V1_ops[3].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V1_ops[3].gen(k_vec=k[n],uc_size=3,uc_pos=1)

    V1 = V1_ops[0]
    for n in range(1,len(V1_ops)):
        V1=H_operations.add(V1,V1_ops[n],np.array([1,1]))

    V2_ops = dict()
    V2_ops[0] = Hamiltonian(pxp,pxp_syms)
    V2_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
    V2_ops[0].model = np.array([[0,0,1,0]])
    V2_ops[0].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V2_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

    V2_ops[1] = Hamiltonian(pxp,pxp_syms)
    V2_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
    V2_ops[1].model = np.array([[0,1,0,0]])
    V2_ops[1].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V2_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=0)

    V2 = V2_ops[0]
    for n in range(1,len(V2_ops)):
        V2=H_operations.add(V2,V2_ops[n],np.array([1,1]))

    V3_ops = dict()
    V3_ops[0] = Hamiltonian(pxp,pxp_syms)
    V3_ops[0].site_ops[1] = np.array([[0,1],[1,0]])
    V3_ops[0].model = np.array([[0,1,1,1,0]])
    V3_ops[0].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V3_ops[0].gen(k_vec=k[n],uc_size=3,uc_pos=0)

    V3_ops[1] = Hamiltonian(pxp,pxp_syms)
    V3_ops[1].site_ops[1] = np.array([[0,1],[1,0]])
    V3_ops[1].model = np.array([[0,1,1,1,0]])
    V3_ops[1].model_coef = np.array([1])
    for n in range(0,np.size(k,axis=0)):
        V3_ops[1].gen(k_vec=k[n],uc_size=3,uc_pos=2)

    V3 = V3_ops[0]
    for n in range(1,len(V3_ops)):
        V3=H_operations.add(V3,V3_ops[n],np.array([1,1]))

    H0 = spin_Hamiltonian(pxp,"x",pxp_syms)
    for n in range(0,np.size(k,axis=0)):
        H0.gen(k[n])

    min_t = 1.5
    d = 0.01
    t=np.arange(-min_t,3,d)
    r = np.zeros(np.size(t))

    pbar=ProgressBar()
    for t_index in pbar(range(0,np.size(t,axis=0))):
        t_val = t[t_index]
        coef = np.array([t_val*0.18243653,-t_val*0.10390499,t_val*0.054452])
        H = H_operations.add(H0,V1,np.array([1,coef[0]]))
        H = H_operations.add(H,V2,np.array([1,coef[1]]))
        H = H_operations.add(H,V3,np.array([1,coef[2]]))

        for n in range(0,np.size(k,axis=0)):
            H.sector.find_eig(k[n])
        ls = level_stats(H.sector.eigvalues(k[0]))
        r[t_index] = ls.mean()
        # print(ls.mean())

    #replace zero manually, need to use parity symmetry
    # zero_index = find_index_bisection(0,t)
    # zero_index = int(min_t/d)
    # pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])
    # z=zm_state(3,1,pxp)
    # k=[0,0]
    # H = spin_Hamiltonian(pxp,"x",pxp_syms)
    # H.gen(k)
    # H.sector.find_eig(k)
    # ls = level_stats(H.sector.eigvalues(k))
    # r[zero_index] = ls.mean()

    np.save("pxp,z3_perts,level_stat_scan,"+str(N),r)
    np.save("pxp,z3_perts,t,"+str(N),t)
    # plt.plot(t,r)
    # plt.axvline(x=1)
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$\langle r \rangle$")
    # plt.title(r"$PXP + t \lambda_i V_i$ Level Stats, N="+str(N))
    # plt.show()
