# import required packages
import os
import gpt as g
import qcd_ml
from qcd_ml.util.solver import GMRES
import torch

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_multigrid_5d import ZPP_Multigrid_5d

# load parameters
pv = snakemake.wildcards.pv
seed = snakemake.params.seed
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
nbasisvectors = snakemake.params.nbasisvectors
block_size = snakemake.params.block_size
solver_kwargs = snakemake.params.solver_kwargs

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op

# initialize random number generator
torch.manual_seed(seed)

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
L5 = [mobius_p['Ls']] + U[0].grid.fdimensions

# domain-wall operator or pauli-villars preconditioned domain wall operator
D = qcdml_op(U, mobius_p)

# multigrid setup and save
src = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
orig_vecs = [torch.randn_like(src) for _ in range(nbasisvectors)]
mg_setup = ZPP_Multigrid_5d.gen_from_fine_vectors(orig_vecs, block_size, lambda b, x0: GMRES(D, b, x0, **solver_kwargs), verbose=True)
mg_setup.save(snakemake.output.multigrid_setup)
