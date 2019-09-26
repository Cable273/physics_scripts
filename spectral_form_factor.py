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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
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
pxp_syms=model_sym_data(pxp,[parity(pxp),translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)
k=[0,0]
H.gen(k)
H.sector.find_eig(k)
e = H.sector.eigvalues(k)
to_del=[]
for n in range(0,np.size(e,axis=0)):
    if np.abs(e[n]) <1e-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    e=np.delete(e,to_del[n])
    

#rectangular billiards
# n_max = 200
# phi = 1/2*(1+np.power(5,0.5))
# e=[]
# for n in range(0,n_max):
    # for m in range(0,n_max):
        # energy = np.power(n/phi,2)+np.power(m,2)
        # e = np.append(e,energy)
# e = np.sort(e)

#random GUE
# dim=4000
# #random gaussian
# M=np.random.normal(0,10,(dim,dim))
# #random hermitian
# M = M + np.transpose(M)
# e,u = np.linalg.eigh(M)

# count number of eigenstates with E_eig<E
N_step = np.arange(0,np.size(e,axis=0))
print("dim="+str(np.size(e)))

def idos_fit(E,a,b,c,d,e,f,g,h,i):
    return a*np.power(E,8)+b*np.power(E,7)+c*np.power(E,6)+d*np.power(E,5)+e*np.power(E,4)+f*np.power(E,3)+g*np.power(E,2)+h*E+i

def dos_fit(E,a,b,c,d,e,f,g,h,i):
    return 8*a*np.power(E,7)+7*b*np.power(E,6)+6*c*np.power(E,5)+5*d*np.power(E,4)+4*e*np.power(E,3)+3*f*np.power(E,2)+2*g*E+h

def idos_poly(E,params):
    powers=np.flip(np.arange(0,np.size(params)))
    temp = 0
    for m in range(0,np.size(powers,axis=0)):
        temp = temp + params[m]*np.power(E,powers[m])
    return temp

def dos_poly(E,params):
    powers=np.flip(np.arange(0,np.size(params)))
    temp = 0
    for m in range(0,np.size(powers,axis=0)-1):
        temp = temp + powers[m]*params[m]*np.power(E,powers[m]-1)
    return temp
    
from scipy.optimize import curve_fit
poly_dim = 12
params = np.polyfit(e,N_step,poly_dim)

N_fit = np.zeros(np.size(e))
dos= np.zeros(np.size(e))
for n in range(0,np.size(e,axis=0)):
    N_fit[n] = idos_poly(e[n],params)
    dos[n] = dos_poly(e[n],params)

plt.plot(e,N_step,label=r"$N(E)$")
plt.plot(e,N_fit,label=r"$N(E) ,\ Fit$")
plt.xlabel(r"$E$")
plt.legend()
plt.title(r"$\textrm{Integrated Density of States}$")
plt.show()

plt.plot(e,dos)
plt.xlabel(r"$E$")
plt.ylabel(r"$\rho(E)$")
plt.title(r"$\textrm{Density of States}$")
plt.show()

#unfold spectrum
unfolded_e = np.zeros(np.size(e))
for n in range(0,np.size(unfolded_e,axis=0)):
    unfolded_e[n] = idos_poly(e[n],params)

params = np.polyfit(unfolded_e,N_step,poly_dim)
N_fit_unfolded = np.zeros(np.size(unfolded_e))
dos_unfolded = np.zeros(np.size(unfolded_e))
for n in range(0,np.size(unfolded_e,axis=0)):
    N_fit_unfolded[n] = idos_poly(unfolded_e[n],params)
    dos_unfolded[n] = dos_poly(unfolded_e[n],params)

plt.plot(unfolded_e,N_fit_unfolded,label=r"$N(E) ,\ Fit$")
plt.xlabel(r"$E$")
plt.title(r"$\textrm{Unfolded Integrated Density of States}$")
plt.legend()
plt.show()

plt.plot(unfolded_e,dos_unfolded)
plt.xlabel(r"$E$")
plt.ylabel(r"$\rho(E)$")
plt.title(r"$\textrm{Unfolded Density of States}$")
plt.show()

def delta_approx(t):
    if np.abs(t)>1e-10:
        return 1/N * (np.sin(N*delta/2*t)/np.sin(delta/2*t))**2
    else:
        return N

rho_bar= 1
# rho_bar = np.mean(dos_unfolded)
delta = 1/rho_bar
Th = 2 * math.pi * rho_bar
N = np.size(e)

t_max = 2
delta_t_exact = 0.0001
t=np.arange(0,t_max,delta_t_exact)
pbar=ProgressBar()
sff = np.zeros(np.size(t))
print("Calculating exact Spectral Form Factor")
for n in pbar(range(0,np.size(t,axis=0))):
    phases = np.exp(1j*unfolded_e*t[n])
    if np.abs(t[n])<Th/2:
        sff[n] = np.abs(np.sum(phases))**2/N-delta_approx(t[n])
    else:
        sff[n] = np.abs(np.sum(phases))**2/N

plt.plot(t,sff)
plt.xlabel(r"$t$")
plt.ylabel(r"$K(t)$")
plt.title(r"$\textrm{Spectral Form Factor (Not averaged)}$")
plt.show()
print(sff)

delta_t_avg = 1
sff_avg = np.zeros(np.size(t))
pbar=ProgressBar()
print("Averaging Spectral Form Factor")
for n in pbar(range(0,np.size(t,axis=0))):
    diff_cap = np.min(np.array([delta_t_avg/2,t[n]-t[0],t[int(np.size(t)-1)]-t[n]]))
    diff_entries = diff_cap / delta_t_exact
    min_index = int(n - diff_entries)
    max_index = int(n + diff_entries)
    sff_window = sff[min_index:max_index]
    if np.size(sff_window)==0:
        sff_avg[n] = sff[n]
    else:
        sff_avg[n] = np.mean(sff_window)

print(sff_avg)
# plt.plot(t,np.log10(sff_avg))
plt.plot(t,sff_avg)
plt.axhline(y=0,linestyle="--")
plt.axhline(y=1,linestyle="--")
plt.title(r"$\textrm{Averaged Spectral Form Factor}, \Delta t =$"+str(delta_t_avg))
plt.xlabel(r"$t$")
plt.ylabel(r"$\bar{K}(t)$")
plt.show()

