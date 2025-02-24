# import required packages
import os
import gpt as g
from gpt.qcd.fermion import wilson_clover
from gpt.algorithms import inverter

# load parameters
seed = snakemake.params.seed
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
split_chiral = snakemake.params.split_chiral
gconfig = snakemake.wildcards.gconfig
nbasisvectors = snakemake.params.nbasisvectors
block_size = snakemake.params.block_size

# initialize random number generator
rng = g.random(seed)

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# define transitions between grids (setup)
def find_near_null_vectors(wc, cgrid):
    slv = inverter.fgmres(eps=1e-3, maxiter=50, restartlen=25, checkres=False)(wc)
    basis = g.orthonormalize(rng.cnormal([wc.vector_space[0].lattice() for _ in range(nbasisvectors)]))
    null = g.lattice(basis[0])
    null[:] = 0
    for b in basis:
        slv(b, null)
    # TODO: apply open boundaries, e.g., in this function
    if split_chiral:
        g.qcd.fermion.coarse.split_chiral(basis)
    bm = g.block.map(cgrid, basis)
    bm.orthonormalize()
    bm.check_orthogonality()
    return basis

# setup multigrid and save
mg_setup_2lvl = inverter.multi_grid_setup(block_size=block_size, projector=find_near_null_vectors)
mg_setup_2lvl_dp = mg_setup_2lvl(D_wc)
g.save(snakemake.output.multigrid_setup, mg_setup_2lvl_dp)
