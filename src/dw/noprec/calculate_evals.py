# import required packages
import os
import gpt as g
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_evals import qcdml_eigvals, qcdml_eigvals_sparse

# load parameters
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
pv = snakemake.wildcards.pv
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
volume = np.prod(L5)

# domain-wall operator or pauli-villars preconditioned domain wall operator
D = qcdml_op(U, mobius_p)

# compute and save eigenvalues
if volume <= 1024:
    print("Computing eigenvalues with dense matrix")
    evals = qcdml_eigvals(L5, D)
else:
    print("Computing eigenvalues with sparse matrix")
    n_evals = 500
    shift = -160 if pv else -20
    evals = qcdml_eigvals_sparse(L5, D, shift=shift, k=n_evals)
np.savetxt(snakemake.output.evals, evals)
