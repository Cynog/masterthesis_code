# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_evals import eigvals
from gpt_models import get_ptc1hxl

# load parameters
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
n_layers = int(snakemake.wildcards.n_layers)

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# load ptc1hxl model from previous training
model = get_ptc1hxl(U, n_layers)
W = g.load(snakemake.input.model_weights)
prec = model(W)

# compute eigenvalues of preconditioned wilson-clover operator
evals = eigvals(grid, prec * D_wc)

# save eigenvalues
np.savetxt(snakemake.output.evals_ptc, evals)
