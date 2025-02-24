# import required packages
import os
import numpy as np
import gpt as g
from gpt.qcd.fermion import wilson_clover
from gpt.algorithms import inverter

# load parameters
seed_solve = snakemake.params.seed_solve
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
n_calciter = snakemake.params.n_calciter
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
fgmres_kwargs = snakemake.params.fgmres_kwargs

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# initialize random number generator
rng = g.random(seed_solve)

# solve for n_calciter times
iterations = []
histories = []
for i in range(n_calciter):
    # random normal right hand side
    src = g.vspincolor(grid)
    src = g.random.cnormal(rng, src)

    # solve without preconditioning
    slv = inverter.fgmres(fgmres_kwargs)
    sol = slv(D_wc)(src)
    
    # save number of iterations and history
    histories.append(slv.history)
    if len(slv.history) == fgmres_kwargs['maxiter']:
        iterations.append(-fgmres_kwargs['maxiter'])
        break
    iterations.append(len(slv.history))

# save number of iterations 
with open(snakemake.output.iterations, "w") as fout:
    for it in iterations:
        fout.write(f"{it}\n")
        
# save history
for i in range(n_calciter):
    if i < len(histories):
        np.savetxt(os.path.join(snakemake.output.history_dir, f"{i}.txt"), histories[i])
    else:
        np.savetxt(os.path.join(snakemake.output.history_dir, f"{i}.txt"), np.array([]))
