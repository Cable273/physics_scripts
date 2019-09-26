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

N = 15
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

V1 = np.zeros((pxp.dim,pxp.dim))
V2 = np.zeros((pxp.dim,pxp.dim))
V3 = np.zeros((pxp.dim,pxp.dim))

for n in range(0,np.size(pxp.basis,axis=0)):
    bits = np.copy(pxp.basis[n])
    for m in range(0,np.size(bits,axis=0)):
        if m == np.size(bits)-1:
            mp1 = 0
            mp2 = 1
            mp3 = 2
            mp4 = 3
        elif m == np.size(bits)-2:
            mp1 = m+1
            mp2 = 0
            mp3 = 1
            mp4 = 2
        elif m == np.size(bits)-3:
            mp1 = m+1
            mp2 = m+2
            mp3 = 0
            mp4 = 1
        elif m == np.size(bits)-4:
            mp1 = m+1
            mp2 = m+2
            mp3 = m+3
            mp4 = 0
        else:
            mp1 = m+1
            mp2 = m+2
            mp3 = m+3
            mp4 = m+4

        #V1
        if m % 3 == 1:
            if bits[m] == 0 and bits[mp2] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp1] == 1:
                    new_bits[mp1] = 0
                else:
                    new_bits[mp1] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V1[pxp.keys[new_ref],n] = V1[pxp.keys[new_ref],n] + 1

            if bits[m] == 0 and bits[mp1] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp2] == 1:
                    new_bits[mp2] = 0
                else:
                    new_bits[mp2] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V1[pxp.keys[new_ref],n] = V1[pxp.keys[new_ref],n] + 1

        elif m % 3 == 2:
            if bits[m] == 0 and bits[mp2] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp1] == 1:
                    new_bits[mp1] = 0
                else:
                    new_bits[mp1] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V1[pxp.keys[new_ref],n] = V1[pxp.keys[new_ref],n] + 1

            if bits[m] == 0 and bits[mp1] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp2] == 1:
                    new_bits[mp2] = 0
                else:
                    new_bits[mp2] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V1[pxp.keys[new_ref],n] = V1[pxp.keys[new_ref],n] + 1

        #V2
        if m % 3 == 0:
            if bits[m] == 0 and bits[mp1] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp2] == 1:
                    new_bits[mp2] = 0
                else:
                    new_bits[mp2] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V2[pxp.keys[new_ref],n] = V2[pxp.keys[new_ref],n] + 1

            if bits[m] == 0 and bits[mp2] == 0 and bits[mp3] == 0:
                new_bits = np.copy(bits)
                if bits[mp1] == 1:
                    new_bits[mp1] = 0
                else:
                    new_bits[mp1] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V2[pxp.keys[new_ref],n] = V2[pxp.keys[new_ref],n] + 1

        #V3
        if m % 3 == 0 or m % 3 == 2:
            if bits[m] == 0 and bits[mp4] == 0:
                new_bits = np.copy(bits)
                if bits[mp1] == 1:
                    new_bits[mp1] = 0
                else:
                    new_bits[mp1] = 1

                if bits[mp2] == 1:
                    new_bits[mp2] = 0
                else:
                    new_bits[mp2] = 1

                if bits[mp3] == 1:
                    new_bits[mp3] = 0
                else:
                    new_bits[mp3] = 1

                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    V3[pxp.keys[new_ref],n] = V3[pxp.keys[new_ref],n] + 1

H0 = spin_Hamiltonian(pxp,"x")
H0.gen()
H0.sector.find_eig()

from scipy.optimize import minimize,minimize_scalar

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(psi_energy,evolved_state))**2
    return -f

def opt_fidelity(coef,plot=False):
    H = H0.sector.matrix()+coef[0]*V1+coef[1]*V2+coef[2]*V3
    # H = H0.sector.matrix()+coef[0]*V3
    e,u = np.linalg.eigh(H)
    psi = zm_state(3,1,pxp)
    psi_energy = np.conj(u[pxp.keys[psi.ref],:])
    res = minimize_scalar(lambda t:fidelity_eval(psi_energy,e,t),method="golden",bracket=(2.5,3.5))
    f_max = fidelity_eval(psi_energy,e,res.x)
    print(coef,-f_max,res.x)
    if plot is True:
        t=np.arange(0,20,0.01)
        f=np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            f[n] = -fidelity_eval(psi_energy,e,t[n])
        # plt.plot(t,f,label="$H_0+\lambda_i V_i$")
        # plt.show()
    # if np.abs(res.x) < 1e-5:
        # f_max = 0
    return f

# res = minimize(lambda coef:opt_fidelity(coef),method="powell",x0=[0.1968067,-0.07472552,0.06522846])
# # # res = minimize(lambda coef:opt_fidelity(coef,plot=True),method="powell",x0=[-0.1,-0.1,-0.1])
# # # res = minimize(lambda coef:opt_fidelity(coef),method="powell",x0=[0.1,0.1,0.1])
# opt_fidelity(res.x,plot=True)
# opt_fidelity([0.2451307,0,0],plot=True)
# opt_fidelity([-0.23457384,0,0],plot=True)

z=zm_state(3,1,pxp)
coef = [0.18243653,-0.10390499,0.0544521]
# coef = [0,0,0]
H = H0.sector.matrix()+coef[0]*V1+coef[1]*V2+coef[2]*V3
e,u = np.linalg.eigh(H)
psi_energy = np.conj(u[pxp.keys[z.ref],:])
eigenvalues = np.copy(e)
overlap = np.log10(np.abs(psi_energy)**2)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_3 \vert E \rangle \vert^2)$")
# plt.title(r"$PXP$ Optimized $Z_3$ Pertubations, N="+str(pxp.N))
plt.title(r"$PXP$ Optimized $Z_3$ Pertubations, N=18")
plt.scatter(eigenvalues,overlap)
plt.show()

t=np.arange(0,5,0.0001)
f = opt_fidelity(coef,plot=True)
for n in range(0,np.size(f,axis=0)):
    if f[n] < 0.1:
        cut = n
        break
index = np.argmax(f[cut:])
print(t[cut:][index],f[cut:][index])
# plt.plot(t,f)
# plt.show()

# # f = fidelity(z,H0).eval(t,z)
# # plt.plot(t,f,label="$H_0$")
# # plt.legend()
# # plt.xlabel(r"$t$")
# # plt.ylabel(r"$\vert \langle Z_3 \vert e^{-iHt} \vert Z_3 \rangle \vert^2$")
# # plt.title(r"$PXP$ Optimized $Z_3$ Pertubations, N="+str(pxp.N))
# # plt.show()
