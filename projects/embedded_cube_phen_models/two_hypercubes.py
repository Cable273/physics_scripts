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

# N = 22
# pxp = unlocking_System([0],"periodic",2,N)
# pxp.gen_basis()
# pxp_syms=model_sym_data(pxp,[translational(pxp)])

# f_m
L_vals = np.arange(500,2000,10)
f_max_vals = np.zeros(np.size(L_vals))
L=10
# L = L_vals[index]
S=L/2
m=np.arange(-S,S)
couplings = np.power(S*(S+1)-m*(m+1),0.5)
H_L = np.diag(couplings,1)+np.diag(couplings,-1)
dim_L = np.size(H_L,axis=0)
dim = 2*np.size(H_L,axis=0)-1
H = np.zeros((dim,dim))
H[:dim_L,:dim_L] = H_L
H[dim_L-1:,dim_L-1:] = H_L

e,u = np.linalg.eigh(H)
plt.plot(e)
plt.show()
# psi = zm_state(2,1,pxp).prod_basis()
# psi_energy= np.conj(u[0,:])
# t=np.arange(0,20,0.01)
# f=np.zeros(np.size(t))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(t,axis=0))):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # #plot wf
    # evolved_state_prod_basis = np.dot(u,evolved_state)
    # plt.plot(np.abs(evolved_state_prod_basis)**2)
    # plt.axhline(y=1)
    # plt.axvline(x=L)
    # plt.tight_layout()
    # plt.xlabel(r"Hamming Distance")
    # plt.ylabel(r"$\vert \psi(n) \vert^2$")
    # plt.title(r"Two Hypercubes, L="+str(L))

    # no_zeros = 5
    # if n != 0:
        # digits = int(np.log10(n))+1
    # else:
        # digits = 1
    # zeros_needed = no_zeros-digits
    # zeros=""
    # if zeros_needed >=1:
        # for m in range(0,zeros_needed):
            # zeros += "0"
    # label = zeros+str(n)

    # plt.savefig("./gif/img,"+label)
    # plt.cla()


    # f[n] = np.abs(np.vdot(psi_energy,evolved_state))**2
# plt.plot(t,f)
# plt.show()

# # for n in range(0,np.size(f,axis=0)):
    # # if f[n]<0.1:
        # # cut = n
        # # break
# # f_max =np.max(f[cut:])
# # f_max_vals[index] = f_max

# # f_max_temp = f_max_vals/L_vals
# # # plt.plot(L_vals,f_max_temp,label="f0/N")
# # # plt.plot(L_vals,1/L_vals,label="1/N")
# # plt.plot(np.log(L_vals),np.log(f_max_vals))
# # plt.title(r"Two Hypercubes, $F_{max}$")
# # plt.xlabel(r"$\log(L)$")
# # plt.ylabel(r"$\log(f0)$")
# # plt.legend()
# # plt.show()

# # # z=zm_state(2,1,pxp)
# # # k=pxp_syms.find_k_ref(z.ref)
# # # H_pxp = spin_Hamiltonian(pxp,"x",pxp_syms)
# # # for n in range(0,np.size(k,axis=0)):
    # # # H_pxp.gen(k[n])
    # # # H_pxp.sector.find_eig(k[n])


# # # fidelity(z,H_pxp,"use sym").plot(t,z)
# # # plt.plot(t,f,label="two cubes")
# # # plt.legend()
# # # plt.show()

# # # print(e)
# # # theta_vals = np.arange(0,2*math.pi+0.01,0.01)
# # # f_max = np.zeros(np.size(theta_vals))
# # # for count in range(0,np.size(theta_vals,axis=0)):
    # # # psi = np.zeros(np.size(e))
    # # # psi[0] = np.cos(theta_vals[count])
    # # # psi[np.size(psi)-1] = np.sin(theta_vals[count])
    # # # print(psi)
    # # # # psi = psi / np.power(np.vdot(psi,psi),0.5)
    # # # psi_energy = np.dot(np.conj(np.transpose(u)),psi)
    # # # # overlap = np.abs(psi_energy)**2

    # # # # plt.scatter(e,overlap)
    # # # # plt.xlabel(r"$E$")
    # # # # plt.ylabel(r"$\log(\vert \langle 0_L \vert E \rangle \vert^2)$")
    # # # # plt.title(r"Two hypercube connected at $\vert 0000.... \rangle$, $\vert 0_L \rangle$ overlap, $L_{hamming}=$"+str(L))
    # # # # plt.show()

    # # # # t=np.arange(0,10,0.01)
    # # # # f=np.zeros(np.size(t))
    # # # # pbar=ProgressBar()
    # # # # for n in pbar(range(0,np.size(t,axis=0))):
        # # # # evolved_state = time_evolve_state(psi_energy,e,t[n])
        # # # # f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    # # # # plt.plot(t,f)
    # # # # plt.xlabel(r"$t$")
    # # # # plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
    # # # # plt.title(r"Two hypercubes, $\theta =$" +str("{0:.2f}".format(theta_vals[count])+" $L=$"+str(L)+"\n"+r"$\vert \psi \rangle = \cos(\theta) \vert 0 \rangle + \sin(\theta) \vert 2L \rangle$"))
    # # # # plt.tight_layout()
    # # # # plt.savefig("./gif/img"+str(count))
    # # # # plt.cla()
    # # # # plt.show()

    # # # def fidelity(t):
        # # # evolved_state = time_evolve_state(psi_energy,e,t)
        # # # f = np.abs(np.vdot(evolved_state,psi_energy))**2
        # # # return -f

    # # # from scipy.optimize import minimize_scalar
    # # # res = minimize_scalar(lambda t: fidelity(t),method="golden",bracket=(4.5,5.5))
    # # # print(res.x)
    # # # f_max[count] = -fidelity(res.x)

# # # plt.xlabel(r"$\theta$")
# # # plt.ylabel(r"$F_{max}$")
# # # plt.title(r"First revival maximum, Two hypercubes, $\vert \psi \rangle = \cos(\theta) \vert 0 \rangle + \sin(\theta) \vert 2L \rangle, L=$"+str(L))
# # # plt.axvline(x=3*math.pi/4,label=r"$\theta = \frac{3 \pi}{4}$")
# # # plt.axvline(x=1*math.pi/4,label=r"$\theta = \frac{\pi}{4}$")
# # # plt.legend()
# # # plt.plot(theta_vals,f_max)
# # # plt.show()

    # # # # for n in range(0,np.size(f,axis=0)):
        # # # # if f[n] < 0.1:
            # # # # cut = n
            # # # # break
    # # # # f_max = np.max(f[cut:])

    # # # # plt.plot(t,f)
    # # # # # plt.title("Two Hypercubes, coupled by common node, \n Symmetric Quench Fidelity, L="+str(L))
    # # # # plt.title("Two Hypercubes, coupled by common node, \n Asymmetric Quench Fidelity, L="+str(L))
    # # # # plt.xlabel(r"$t$")
    # # # # plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \vert \rangle^2$")
    # # # # plt.show()
