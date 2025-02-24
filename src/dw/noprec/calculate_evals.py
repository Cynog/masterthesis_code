# import required packages
import os
import gpt as g
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_evals import qcdml_eigvals

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

# domain-wall operator or pauli-villars preconditioned domain wall operator
D = qcdml_op(U, mobius_p)

# compute and save eigenvalues
evals = qcdml_eigvals(L5, D)
np.savetxt(snakemake.output.evals, evals)
