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

N = 6
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

pert_coef = np.array([0.051,0.0102,3.1875e-3,1.1333e-3])
# pert_coef = 0.051
# H0=spin_Hamiltonian(pxp,"x",pxp_syms)
# V = Hamiltonian(pxp,pxp_syms)
# V.site_ops[1] = np.array([[0,1],[1,0]])
# V.site_ops[2] = np.array([[-1,0],[0,1]])
# V.model = np.array([[2,0,1,0],[0,1,0,2]])
# V.model_coef = np.array([1,1])
# V.gen()
# H0.gen()
# H=H_operations.add(H0,V,np.array([1,-pert_coef]))

H=Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[-1,0],[0,1]])
H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0],[2,-1,0,1,0],[0,1,0,-1,2],[2,-1,-1,0,1,0],[0,1,0,-1,-1,2],[2,-1,-1,-1,0,1,0],[0,1,0,-1,-1,-1,2]])
H.model_coef = np.array([1,-pert_coef[0],-pert_coef[0],-pert_coef[1],-pert_coef[1],-pert_coef[2],-pert_coef[2],-pert_coef[3],-pert_coef[3]])

# H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0]])
# H.model_coef = np.array([1,-pert_coef[0],-pert_coef[0]])

H.gen()

H.sector.find_eig()
z=zm_state(2,1,pxp)
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()

#find rescaling such that E1-E0 = 1
def gap(H,c):
    e,u = np.linalg.eigh(c*H)
    gap = e[1]-e[0]
    return np.abs(1-gap)

from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda c: gap(H.sector.matrix(),c)-1,method="golden", bracket=(0.9,1.1))
c=res.x
print(c)

H=Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[-1,0],[0,1]])
H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0],[2,-1,0,1,0],[0,1,0,-1,2],[2,-1,-1,0,1,0],[0,1,0,-1,-1,2],[2,-1,-1,-1,0,1,0],[0,1,0,-1,-1,-1,2]])
H.model_coef = np.array([c,-c*pert_coef[0],-c*pert_coef[0],-c*pert_coef[1],-c*pert_coef[1],-c*pert_coef[2],-c*pert_coef[2],-c*pert_coef[3],-c*pert_coef[3]])
# H.model = np.array([[0,1,0],[0,1,0,2],[2,0,1,0]])
# H.model_coef = np.array([c,-c*pert_coef[0],-c*pert_coef[0]])
H.gen()
H.sector.find_eig()

z=zm_state(2,1,pxp)
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()

overlap = eig_overlap(z,H).eval()
from Calculations import get_top_band_indices
scar_indices = np.sort(np.unique(get_top_band_indices(H.sector.eigvalues(),overlap,N)))

scar_basis = np.zeros((pxp.dim,np.size(scar_indices)))
for n in range(0,np.size(scar_indices,axis=0)):
    scar_basis[scar_indices[n],n] = 1
    

plt.scatter(H.sector.eigvalues(),overlap)
for n in range(0,np.size(scar_indices,axis=0)):
    plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],s=100,marker="x",color="red")
plt.show()

# H=H_operations.add(H0,V,np.array([c,-c*pert_coef]))
# H.sector.find_eig()
# print(H.sector.eigvalues()[1]-H.sector.eigvalues()[0])

#P+P on even sites
pe = Hamiltonian(pxp,pxp_syms)
pe.site_ops[1] = np.array([[0,1],[0,0]])
pe.model = np.array([[0,1,0]])
pe.model_coef = np.array([1])
pe.gen(parity=1)

#ZP+P, + on even sites
zPpPe = Hamiltonian(pxp,pxp_syms)
zPpPe.site_ops[1] = np.array([[0,1],[0,0]])
zPpPe.site_ops[2] = np.array([[-1,0],[0,1]])
zPpPe.model = np.array([[2,0,1,0]])
zPpPe.model_coef = np.array([1])
zPpPe.gen(parity=0)

#P+PZ, + on even sites
PpPze = Hamiltonian(pxp,pxp_syms)
PpPze.site_ops[1] = np.array([[0,1],[0,0]])
PpPze.site_ops[2] = np.array([[-1,0],[0,1]])
PpPze.model = np.array([[0,1,0,2]])
PpPze.model_coef = np.array([1])
PpPze.gen(parity=1)

# #ZIP+P, + on even sites
ziPpPe = Hamiltonian(pxp,pxp_syms)
ziPpPe.site_ops[1] = np.array([[0,1],[0,0]])
ziPpPe.site_ops[2] = np.array([[-1,0],[0,1]])
ziPpPe.model = np.array([[2,-1,0,1,0]])
ziPpPe.model_coef = np.array([1])
ziPpPe.gen(parity=1)

#P+PIZ, + on even sites
PpPize = Hamiltonian(pxp,pxp_syms)
PpPize.site_ops[1] = np.array([[0,1],[0,0]])
PpPize.site_ops[2] = np.array([[-1,0],[0,1]])
PpPize.model = np.array([[0,1,0,-1,2]])
PpPize.model_coef = np.array([1])
PpPize.gen(parity=1)

#ZIIP+P, + on even sites
ziiPpPe = Hamiltonian(pxp,pxp_syms)
ziiPpPe.site_ops[1] = np.array([[0,1],[0,0]])
ziiPpPe.site_ops[2] = np.array([[-1,0],[0,1]])
ziiPpPe.model = np.array([[2,-1,-1,0,1,0]])
ziiPpPe.model_coef = np.array([1])
ziiPpPe.gen(parity=0)

#P+PIIZ, + on even sites
PpPiize = Hamiltonian(pxp,pxp_syms)
PpPiize.site_ops[1] = np.array([[0,1],[0,0]])
PpPiize.site_ops[2] = np.array([[-1,0],[0,1]])
PpPiize.model = np.array([[0,1,0,-1,-1,2]])
PpPiize.model_coef = np.array([1])
PpPiize.gen(parity=1)

#ZIIIP+P, + on even sites
ziiiPpPe = Hamiltonian(pxp,pxp_syms)
ziiiPpPe.site_ops[1] = np.array([[0,1],[0,0]])
ziiiPpPe.site_ops[2] = np.array([[-1,0],[0,1]])
ziiiPpPe.model = np.array([[2,-1,-1,-1,0,1,0]])
ziiiPpPe.model_coef = np.array([1])
ziiiPpPe.gen(parity=1)

#P+PIIIZ, + on even sites
PpPiiize = Hamiltonian(pxp,pxp_syms)
PpPiiize.site_ops[1] = np.array([[0,1],[0,0]])
PpPiiize.site_ops[2] = np.array([[-1,0],[0,1]])
PpPiiize.model = np.array([[0,1,0,-1,-1,-1,2]])
PpPiiize.model_coef = np.array([1])
PpPiiize.gen(parity=1)

#P-P on odd sites
mo = Hamiltonian(pxp,pxp_syms)
mo.site_ops[1] = np.array([[0,0],[1,0]])
mo.model = np.array([[0,1,0]])
mo.model_coef = np.array([1])
mo.gen(parity=0)

#ZP-P, - on odd sites
zPmPo = Hamiltonian(pxp,pxp_syms)
zPmPo.site_ops[1] = np.array([[0,0],[1,0]])
zPmPo.site_ops[2] = np.array([[-1,0],[0,1]])
zPmPo.model = np.array([[2,0,1,0]])
zPmPo.model_coef = np.array([1])
zPmPo.gen(parity=1)

# #P-PZ, - on odd sites
PmPzo = Hamiltonian(pxp,pxp_syms)
PmPzo.site_ops[1] = np.array([[0,0],[1,0]])
PmPzo.site_ops[2] = np.array([[-1,0],[0,1]])
PmPzo.model = np.array([[0,1,0,2]])
PmPzo.model_coef = np.array([1])
PmPzo.gen(parity=0)

# #ZIP-P, - on odd sites
ziPmPo = Hamiltonian(pxp,pxp_syms)
ziPmPo.site_ops[1] = np.array([[0,0],[1,0]])
ziPmPo.site_ops[2] = np.array([[-1,0],[0,1]])
ziPmPo.model = np.array([[2,-1,0,1,0]])
ziPmPo.model_coef = np.array([1])
ziPmPo.gen(parity=0)

#P-PIZ, - on odd sites
PmPizo = Hamiltonian(pxp,pxp_syms)
PmPizo.site_ops[1] = np.array([[0,0],[1,0]])
PmPizo.site_ops[2] = np.array([[-1,0],[0,1]])
PmPizo.model = np.array([[0,1,0,-1,2]])
PmPizo.model_coef = np.array([1])
PmPizo.gen(parity=0)

#ZIIP-P, - on odd sites
ziiPmPo = Hamiltonian(pxp,pxp_syms)
ziiPmPo.site_ops[1] = np.array([[0,0],[1,0]])
ziiPmPo.site_ops[2] = np.array([[-1,0],[0,1]])
ziiPmPo.model = np.array([[2,-1,-1,0,1,0]])
ziiPmPo.model_coef = np.array([1])
ziiPmPo.gen(parity=1)

#P-PIIZ, - on odd sites
PmPiizo = Hamiltonian(pxp,pxp_syms)
PmPiizo.site_ops[1] = np.array([[0,0],[1,0]])
PmPiizo.site_ops[2] = np.array([[-1,0],[0,1]])
PmPiizo.model = np.array([[0,1,0,-1,-1,2]])
PmPiizo.model_coef = np.array([1])
PmPiizo.gen(parity=0)

#ZIIIP-P, - on odd sites
ziiiPmPo = Hamiltonian(pxp,pxp_syms)
ziiiPmPo.site_ops[1] = np.array([[0,0],[1,0]])
ziiiPmPo.site_ops[2] = np.array([[-1,0],[0,1]])
ziiiPmPo.model = np.array([[2,-1,-1,-1,0,1,0]])
ziiiPmPo.model_coef = np.array([1])
ziiiPmPo.gen(parity=0)

#P-PIIIZ, - on odd sites
PmPiiizo = Hamiltonian(pxp,pxp_syms)
PmPiiizo.site_ops[1] = np.array([[0,0],[1,0]])
PmPiiizo.site_ops[2] = np.array([[-1,0],[0,1]])
PmPiiizo.model = np.array([[0,1,0,-1,-1,-1,2]])
PmPiiizo.model_coef = np.array([1])
PmPiiizo.gen(parity=0)

#check hams working right
# temp0 = ref_state(1,pxp)
# temp1 = zm_state(2,1,pxp)
# test0 = np.dot(pe.sector.matrix(),temp0.prod_basis())
# test1 = np.dot(PpPze.sector.matrix(),temp0.prod_basis())
# test2= np.dot(zPpPe.sector.matrix(),temp0.prod_basis())
# test3 = np.dot(mo.sector.matrix(),temp1.prod_basis())
# test4 = np.dot(zPmPo.sector.matrix(),temp1.prod_basis())
# test5= np.dot(PmPzo.sector.matrix(),temp1.prod_basis())
# from Diagnostics import print_wf
# print(temp0.bits)
# print("Pe")
# print_wf(test0,pxp,1e-1)
# print("\n")
# print("PpPze")
# print_wf(test1,pxp,1e-1)
# print("\n")
# print("zPpPe")
# print_wf(test2,pxp,1e-1)
# print("\n")
# print(temp1.bits)
# print("mo")
# print_wf(test3,pxp,1e-1)
# print("\n")
# print("zPmPo")
# print_wf(test4,pxp,1e-1)
# print("\n")
# print("PmPzo")
# print_wf(test5,pxp,1e-1)

Hp = H_operations.add(pe,mo,np.array([c,c]))

Hp = H_operations.add(Hp,zPpPe,np.array([1,-c*pert_coef[0]]))
Hp = H_operations.add(Hp,PpPze,np.array([1,-c*pert_coef[0]]))
Hp = H_operations.add(Hp,zPmPo,np.array([1,-c*pert_coef[0]]))
Hp = H_operations.add(Hp,PmPzo,np.array([1,-c*pert_coef[0]]))

Hp = H_operations.add(Hp,ziPpPe,np.array([1,-c*pert_coef[1]]))
Hp = H_operations.add(Hp,PpPize,np.array([1,-c*pert_coef[1]]))
Hp = H_operations.add(Hp,ziPmPo,np.array([1,-c*pert_coef[1]]))
Hp = H_operations.add(Hp,PmPizo,np.array([1,-c*pert_coef[1]]))

Hp = H_operations.add(Hp,ziiPpPe,np.array([1,-c*pert_coef[2]]))
Hp = H_operations.add(Hp,PpPiize,np.array([1,-c*pert_coef[2]]))
Hp = H_operations.add(Hp,ziiPmPo,np.array([1,-c*pert_coef[2]]))
Hp = H_operations.add(Hp,PmPiizo,np.array([1,-c*pert_coef[2]]))

Hp = H_operations.add(Hp,ziiiPpPe,np.array([1,-c*pert_coef[3]]))
Hp = H_operations.add(Hp,PpPiiize,np.array([1,-c*pert_coef[3]]))
Hp = H_operations.add(Hp,ziiiPmPo,np.array([1,-c*pert_coef[3]]))
Hp = H_operations.add(Hp,PmPiiizo,np.array([1,-c*pert_coef[3]]))

Hp = Hp.sector.matrix()
Hm = np.conj(np.transpose(Hp))

#check H = H+ + H-
print((np.abs(Hp+Hm-H.sector.matrix())<1e-5).all())

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 2*com(Hp,Hm)
plt.matshow(np.abs(Hz))
plt.show()

z=zm_state(2,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,pxp.N):
    new_state = np.dot(Hp,current_state)
    new_state = new_state / np.power(np.vdot(new_state,new_state),0.5)
    fsa_basis = np.vstack((fsa_basis,new_state))
    current_state = new_state

fsa_basis = np.transpose(fsa_basis)
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
Hz_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(Hz,fsa_basis))
e,u = np.linalg.eigh(H_fsa)

print("\n")
print("Hz")
print(np.diag(Hz_fsa))
print("\n")

Z_eff = np.dot(np.conj(np.transpose(scar_basis)),np.dot(np.diag(H.sector.eigvalues()),scar_basis))
print("Z_eff")
print(np.diag(Z_eff))
print("\n")

print("H proj")
print(e)


plt.matshow(np.abs(Hz_fsa))
plt.matshow(np.abs(H_fsa))
plt.show()

Ux_eig = u
P = scar_basis
U_eig = H.sector.eigvectors()


U_fsa = np.dot(P,np.conj(np.transpose(Ux_eig)))
U_fsa = np.dot(U_eig,U_fsa)

z0=U_fsa[:,0]
z0_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),z0)
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z0_energy,H.sector.eigvalues(),t[n])
    f[n] = np.abs(np.vdot(evolved_state,z0_energy))**2
plt.plot(t,f)
plt.show()

print(np.shape(U_fsa))

for n in range(0,np.size(U_fsa,axis=1)):
    for m in range(0,np.size(U_fsa,axis=1)):
        temp = np.abs(np.vdot(U_fsa[:,n],U_fsa[:,m]))
        if temp>1e-5:
            print(temp,n,m)

temp = np.dot(np.conj(np.transpose(U_fsa)),np.dot(H.sector.matrix(),U_fsa))
temp2 = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))

print("IS true??")
print((np.abs(temp-temp2)<1e-10).all())

print("IHIHIHIHI")
print(np.diag(temp,1))
print(np.diag(temp2,1))
e0,u0 = np.linalg.eigh(temp)
e1,u1 = np.linalg.eigh(temp2)
print("\n")
print(e0)
print(e1)

# e,u = np.linalg.eigh(temp)

# print("HOIHDOISHGOISDHg")
# print(e)

# plt.matshow(np.abs(temp))
# plt.show()
# print(np.shape(U_fsa))
# print(np.shape(fsa_basis))
# print(U_fsa[:,0])
# print(fsa_basis[:,0])

print("\n")
z_prod_basis = zm_state(2,1,pxp).prod_basis()
from Diagnostics import print_wf
for n in range(0,np.size(U_fsa,axis=1)):
    # print("\n")
    # print_wf(U_fsa[:,n],pxp,1e-2)
    overlap = np.abs(np.vdot(z_prod_basis,U_fsa[:,n]))
    # overlap = np.abs(np.vdot(z_prod_basis,fsa_basis[:,n]))
    # print(np.vdot(U_fsa[:,n],U_fsa[:,n]))
    print(overlap)
