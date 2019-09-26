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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state,get_top_band_indices

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0=spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()

Hpe = Hamiltonian(pxp,pxp_syms)
Hpe.site_ops[1] = np.array([[0,1],[0,0]])
Hpe.model = np.array([[0,1,0]])
Hpe.model_coef = np.array([1])
Hpe.gen(parity=0)

Hpo = Hamiltonian(pxp,pxp_syms)
Hpo.site_ops[1] = np.array([[0,1],[0,0]])
Hpo.model = np.array([[0,1,0]])
Hpo.model_coef = np.array([1])
Hpo.gen(parity=1)

Hme = Hamiltonian(pxp,pxp_syms)
Hme.site_ops[1] = np.array([[0,0],[1,0]])
Hme.model = np.array([[0,1,0]])
Hme.model_coef = np.array([1])
Hme.gen(parity=0)

Hmo = Hamiltonian(pxp,pxp_syms)
Hmo.site_ops[1] = np.array([[0,0],[1,0]])
Hmo.model = np.array([[0,1,0]])
Hmo.model_coef = np.array([1])
Hmo.gen(parity=1)

Hp = H_operations.add(Hpe,Hmo,np.array([1,1]))
Hm = H_operations.add(Hme,Hpo,np.array([1,1]))

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = com(Hp.sector.matrix(),Hm.sector.matrix())

z=zm_state(2,1,pxp)
z_prod_basis = z.prod_basis()
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,pxp.N):
    new_state = np.dot(Hp.sector.matrix(),current_state)
    new_state = new_state / np.power(np.vdot(new_state,new_state),0.5)
    fsa_basis = np.vstack((fsa_basis,new_state))
    current_state = new_state

fsa_basis = np.transpose(fsa_basis)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H0.sector.matrix(),fsa_basis))
e_fsa,u_fsa = np.linalg.eigh(H_fsa)

H0.sector.find_eig()
t=np.arange(0,35,0.01)
f_exact = fidelity(z,H0).eval(t,z)

f_fsa = np.zeros(np.size(t))
z_energy_fsa = np.conj(u_fsa[0,:])

for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z_energy_fsa,e_fsa,t[n])
    f_fsa[n] = np.abs(np.vdot(evolved_state,z_energy_fsa))
plt.plot(t,f_exact,label="Exact Evolution")
plt.plot(t,f_fsa,label="FSA Projected Evolution")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.legend()
plt.title(r"$PXP$ Projected time evolution, N="+str(pxp.N))
plt.show()

# svd_basis = np.load("./svd_basis_for_gif/svd_basis,16,150.npy")
# t_info = np.load("./svd_basis_for_gif/t_info,16,150.npy")
# print(t_info[1])

# z_svd = np.dot(np.conj(np.transpose(svd_basis)),z.prod_basis())
# H_svd = np.dot(np.conj(np.transpose(svd_basis)),np.dot(H0.sector.matrix(),svd_basis))
# e_svd,u_svd = np.linalg.eigh(H_svd)

# overlap_svd = np.zeros(np.size(e_svd))
# for n in range(0,np.size(overlap_svd,axis=0)):
    # overlap_svd[n] = np.log10(np.abs(np.vdot(z_svd,u_svd[:,n]))**2)

exact_overlap = eig_overlap(z,H0).eval()
eigenvalues = H0.sector.eigvalues()
to_del=[]
for n in range(0,np.size(exact_overlap,axis=0)):
    if exact_overlap[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    exact_overlap=np.delete(exact_overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])

overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)


    
# plt.scatter(eigenvalues,exact_overlap,label="Exact")
# plt.scatter(e_fsa,overlap_fsa,marker="x",s=100,label="FSA")
# plt.scatter(e_svd,overlap_svd,marker="s",color="red",s=100,alpha=0.6,label="SVD")
# plt.legend()
# plt.show()

# svd projection for comparison
pbar=ProgressBar()
for n in pbar(range(1,349,1)):
    svd_basis = np.load("./svd_basis_for_gif/svd_basis,16,"+str(n)+".npy")
    t_info = np.load("./svd_basis_for_gif/t_info,16,"+str(n)+".npy")

    z_svd = np.dot(np.conj(np.transpose(svd_basis)),z_prod_basis)
    H_svd = np.dot(np.conj(np.transpose(svd_basis)),np.dot(H0.sector.matrix(),svd_basis))
    e_svd,u_svd = np.linalg.eigh(H_svd)

    overlap_svd = np.zeros(np.size(e_svd))
    for m in range(0,np.size(overlap_svd,axis=0)):
        overlap_svd[m] = np.log10(np.abs(np.vdot(z_svd,u_svd[:,m]))**2)

    total_digits = 6
    no_digits = int(np.log10(n)+1)
    zeros_needed = total_digits-no_digits
    if zeros_needed>0:
        zeros = "0"
        for m in range(0,zeros_needed):
            zeros+="0"
            label=zeros+str(n)
    else:
        label=str(n)
    # print(label)

    plt.scatter(eigenvalues,exact_overlap,label="Exact")
    plt.scatter(e_fsa,overlap_fsa,marker="x",s=100,label="FSA")
    plt.scatter(e_svd,overlap_svd,marker="s",color="red",s=100,alpha=0.6,label="SVD")
    # plt.legend()
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\log(\vert \langle Z_2 \vert E \rangle \vert^2)$")
    plt.title(r"$PXP$, SVD tol=0.1L, N=16"+"\n"+"$t_{max}=$"+str("{0:.2f}".format(t_info[0]))+r", $N_{kept}=$"+str(t_info[1]))
    plt.tight_layout()
    # plt.show()
    plt.savefig("./gif/img"+label)
    plt.cla()



    # z_svd_energy = np.dot(np.conj(np.transpose(u_svd)),z_svd)
    # f_svd = np.zeros(np.size(t))
    # for m in range(0,np.size(t,axis=0)):
        # evolved_state = time_evolve_state(z_svd_energy,e_svd,t[m])
        # f_svd[m] = np.abs(np.vdot(evolved_state,z_svd_energy))**2



    # plt.plot(t,f_svd,linestyle="--",label="SVD Projected Evolution")
    # plt.plot(t,f_exact,alpha=0.8,label="Exact Evolution")
    # plt.plot(t,f_fsa,alpha=0.8,label="FSA Projected Evolution")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
    # plt.title(r"$PXP$, SVD tol=0.01L, N=16"+"\n"+"$t_{max}=$"+str("{0:.2f}".format(t_info[0]))+r", $N_{kept}=$"+str(t_info[1]))
    # # plt.legend()
    # plt.tight_layout()
    # plt.savefig("./gif/img"+label)
    # plt.cla()
    # # plt.show()

