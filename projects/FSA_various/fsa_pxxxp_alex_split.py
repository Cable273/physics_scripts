#!/usr/bin/env python# -*- coding: utf-8 -*-

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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state, gram_schmidt,gen_krylov_basis

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 16
base=2
pxp = unlocking_System([0],"periodic",base,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

z=zm_state(2,1,pxp)

#krylov basis
krylov_dim = pxp.N
krylov_basis = gen_krylov_basis(H.sector.matrix(),krylov_dim,z.prod_basis(),pxp,orth="qr")

#FSA basis
# P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.model = np.array([[0,1,0],])
pe.model_coef = np.array([1])
pe.gen(parity=1)
#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=0)

Hp = pe.sector.matrix()+mo.sector.matrix()
Hm = np.conj(np.transpose(Hp))

# a = 0.51
a_vals = np.arange(0.5,1.1,0.1)
for count in range(0,np.size(a_vals,axis=0)):
    a=a_vals[count]
    Hp_new = a * Hp + (1-a)* Hm
    Hm_new = np.conj(np.transpose(Hp_new))
    print((np.abs(Hp_new-Hm_new)<1e-5).all())

    #Raising op
    def com(a,b):
        return np.dot(a,b)-np.dot(b,a)
    Hz = 1/2*com(Hp_new,Hm_new)
    e,u = np.linalg.eigh(Hz)
    from Diagnostics import print_wf
    print_wf(u[:,0],pxp,1e-2)

    fsa_basis = z.prod_basis()
    current_state = fsa_basis
    for n in range(0,krylov_dim):
        next_state = np.dot(Hp_new,current_state)
        next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
        fsa_basis = np.vstack((fsa_basis,next_state))
        current_state = next_state
    fsa_basis = np.transpose(fsa_basis)
    # fsa_basis,temp = np.linalg.qr(fsa_basis)
    gs = gram_schmidt(fsa_basis)
    gs.ortho()
    fsa_basis = gs.ortho_basis

    #project hamiltonians
    H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
    H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
    Hz_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hz,fsa_basis))
    # plt.matshow(np.abs(H_fsa))
    # plt.matshow(np.abs(Hz_fsa))
    # plt.show()

    e_krylov,u_krylov = np.linalg.eigh(H_krylov)
    e_fsa,u_fsa = np.linalg.eigh(H_fsa)
    print(e_krylov)
    print(e_fsa)

    overlap_krylov = np.log10(np.abs(u_krylov[0,:])**2)
    overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)

    # eig_overlap(z,H).plot()
    # plt.scatter(e_krylov,overlap_krylov,marker="x",s=200,color="cyan",label="Krylov",alpha=0.6)
    # plt.scatter(e_fsa,overlap_fsa,marker="s",s=100,label="FSA",color="red",alpha=0.6)
    # plt.xlabel(r"$E$")
    # plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
    # plt.title(r"$N_c=$"+str(pxp.base)+r", $N=$"+str(pxp.N))
    # plt.legend()
    # plt.show()

    u_krylov_comp = np.dot(krylov_basis,u_krylov)
    u_fsa_comp = np.dot(fsa_basis,u_fsa)

    exact_overlap_krylov = np.zeros(np.size(e_krylov))
    exact_overlap_fsa = np.zeros(np.size(e_fsa))
    for n in range(0,np.size(u_fsa_comp,axis=1)):
        max_overlap_krylov = 0
        max_overlap_fsa = 0
        for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
            temp_krylov = np.abs(np.vdot(u_krylov_comp[:,n],H.sector.eigvectors()[:,m]))
            temp_fsa = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))
            if temp_krylov> max_overlap_krylov:
                max_overlap_krylov = temp_krylov
            if temp_fsa> max_overlap_fsa:
                max_overlap_fsa = temp_fsa
        exact_overlap_krylov[n] = max_overlap_krylov
        exact_overlap_fsa[n] = max_overlap_fsa

    plt.plot(e_fsa,exact_overlap_fsa,label="FSA")
    plt.plot(e_krylov,exact_overlap_krylov,linestyle="--",label="Krylov")
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
    plt.title(r"Deformed FSA, $\alpha=$"+str("{0:.2f}".format(a))+"\n"+r"$N=$"+str(pxp.N))
    # plt.legend()
    plt.tight_layout()
    plt.savefig("../gif/img"+str(count))
    plt.cla()
    # plt.show()
