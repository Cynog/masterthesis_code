# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_evals import eigvals
from gpt_models import get_smoother, get_matrix_operator_smoother

# load parameters
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# load smoother model from previous training
model = get_smoother(U)
W = g.load(snakemake.input.model_weights)
prec = get_matrix_operator_smoother(model, W, grid)

# compute eigenvalues of preconditioned wilson-clover operator
evals = eigvals(grid, prec * D_wc)

# save eigenvalues
np.savetxt(snakemake.output.evals_smoother, evals)
