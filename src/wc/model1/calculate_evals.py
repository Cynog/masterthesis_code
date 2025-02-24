# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_evals import eigvals
from gpt_models import get_model1

# load parameters
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# wison clover operator
wc = wilson_clover(U, fermion_p)

# load model1 from previous training
multigrid_setup_2lvl = g.load(snakemake.input.multigrid_setup)
model = get_model1(U, multigrid_setup_2lvl)
W = g.load(snakemake.input.model_weights)
prec = model(W)

# compute eigenvalues of preconditioned wilson-clover operator
evals = eigvals(grid, prec * wc)

# save eigenvalues
np.savetxt(snakemake.output.evals_model1, evals)
