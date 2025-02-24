# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
import numpy as np

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_evals import eigvals, create_copy_plan_coarse
from gpt_models import get_wc_coarse

# load parameters
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
nbasisvectors = snakemake.params.nbasisvectors
split_chiral = snakemake.params.split_chiral
gconfig = snakemake.wildcards.gconfig

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# load multigrid to get coarse wilson-clover operator
mg_setup_2lvl = g.load(snakemake.input.multigrid_setup)
D_wc_coarse, grid_coarse = get_wc_coarse(D_wc, mg_setup_2lvl)

# compute eigenvalues of coarse wilson-clover operator
nbasisvectors = 2 * nbasisvectors if split_chiral else nbasisvectors
copy_plan = create_copy_plan_coarse(grid_coarse, nbasisvectors)
N = grid_coarse.fsites * nbasisvectors
evals = eigvals(grid_coarse, D_wc_coarse, N, copy_plan)

# save eigenvalues
np.savetxt(snakemake.output.evals_coarse, evals)
