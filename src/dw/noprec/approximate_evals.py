# import required packages
import os
import gpt as g
from gpt.algorithms import inverter
from gpt.qcd.fermion import preconditioner
import numpy as np
import scipy

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_convert import ndarray2lattice_5d, lattice2ndarray_5d

# load parameters
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
pv = snakemake.wildcards.pv
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
n_evals = snakemake.params.n_evals
tol = snakemake.params.tol
maxiter = snakemake.params.maxiter
solver_kwargs = snakemake.params.solver_kwargs
tol_op = snakemake.params.tol_op

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
dim = np.prod(L5) * 4 * 3
    
# domain-wall operator or pauli-villars preconditioned domain wall operator
D_dw = g.qcd.fermion.mobius(U, mobius_p)
grid5 = D_dw.F_grid
if pv == "":
    D = D_dw
else:
    pv_p = mobius_p.copy()
    pv_p["mass"] = 1.0
    D_pv_dag = g.adj(g.qcd.fermion.mobius(U, pv_p))
    D = D_pv_dag * D_dw
    
def D_np(v_np):
    v_np = v_np.reshape(*L5, 4, 3).astype(np.complex128)
    v_gpt = ndarray2lattice_5d(v_np, grid5, g.vspincolor)
    w_gpt = D(v_gpt)
    w_np = lattice2ndarray_5d(w_gpt)
    w_np = w_np.reshape(dim)
    return w_np
D_scipy = scipy.sparse.linalg.LinearOperator(shape=(dim, dim), matvec=D_np)

# cg with shur complement preconditioner for the inverse
cg = inverter.cg(solver_kwargs)
prec = preconditioner.eo2_ne(parity=g.odd)
slv = inverter.preconditioned(prec, cg)
D_dw_inv = slv(D_dw)
if pv == "":
    D_inv = D_dw_inv
else:
    slv_pv = inverter.preconditioned(prec, cg)
    D_pv_dag_inv = slv_pv(D_pv_dag)
    D_inv = D_dw_inv * D_pv_dag_inv
    
def D_inv_np(v_np):
    v_np = v_np.reshape(*L5, 4, 3).astype(np.complex128)
    v_gpt = ndarray2lattice_5d(v_np, grid5, g.vspincolor)
    w_gpt = D_inv(v_gpt)
    w_np = lattice2ndarray_5d(w_gpt)
    w_np = w_np.reshape(dim)
    return w_np
D_inv_scipy = scipy.sparse.linalg.LinearOperator(shape=(dim, dim), matvec=D_inv_np)

# check that inverse operator works
src = np.random.randn(dim)
dst = D_inv_scipy.matvec(D_scipy.matvec(src))
norm = np.linalg.norm(dst - src)
print(norm)
assert norm < tol_op

# compute and save eigenvalues
evals = scipy.sparse.linalg.eigs(D_scipy, k=n_evals, maxiter=maxiter, tol=tol, return_eigenvectors=False, sigma=0, OPinv=D_inv_scipy)
np.savetxt(snakemake.output.evals, evals)
