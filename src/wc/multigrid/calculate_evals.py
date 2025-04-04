# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_evals import eigvals
from gpt_models import get_mg_2lvl_vcycle, get_matrix_operator_from_solver

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

# load multigrid
mg_setup_2lvl = g.load(snakemake.input.multigrid_setup)
mg_2lvl_vcycle = get_mg_2lvl_vcycle(mg_setup_2lvl)
mg_solver = mg_2lvl_vcycle(D_wc)
prec = get_matrix_operator_from_solver(mg_solver)

# compute eigenvalues of preconditioned wilson-clover operator
evals = eigvals(grid, prec * D_wc)

# save eigenvalues
np.savetxt(snakemake.output.evals_mg, evals)
