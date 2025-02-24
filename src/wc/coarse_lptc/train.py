# import required packages
import os
import numpy as np
import gpt as g
from gpt.qcd.fermion import wilson_clover

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_adam import adam, ADAM_State
from gpt_models import get_coarse_lptc

# load parameters
seed = snakemake.params.seed
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
Wscale = snakemake.params.Wscale
adam_kwargs = snakemake.params.adam_kwargs
ntrainvectors = snakemake.params.ntrainvectors
saveweightsstart = snakemake.params.saveweightsstart
saveweightsevery = snakemake.params.saveweightsevery

# initialize random number generator
rng = g.random(seed)

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
grid = U[0].grid

# define vector
src = g.vspincolor(grid)

# wilson-clover operator
D_wc = wilson_clover(U, fermion_p)

# load multigrid setup
mg_setup_2lvl = g.load(snakemake.input.multigrid_setup)
coarse_grid = mg_setup_2lvl[0][0]
u_bar = mg_setup_2lvl[0][1]
b = g.block.map(coarse_grid, u_bar)
D_wc_coarse = b.coarse_operator(D_wc)
src_coarse = b.project(src)

# local parallel transport convolution model with random weights
model = get_coarse_lptc(mg_setup_2lvl)
W = model.random_weights(rng)
for weight in W:
    weight *= Wscale
    
# initialize to identity
w = W[0]
w += g.identity(w)

# train model
state = None
costs = []
for k in range(1, ntrainvectors+1):
    # generate training data
    source = g.random.cnormal(rng, src_coarse)
    training_outputs = [source]
    training_inputs = [g(D_wc_coarse * s) for s in training_outputs] 
    normalizations = [g.norm2(inp) for inp in training_inputs]
    training_inputs = [g(inp / norm**0.5) for inp,norm in zip(training_inputs, normalizations)]
    training_outputs = [g(oup / norm**0.5) for oup,norm in zip(training_outputs, normalizations)]
    
    # optimize
    cost = model.cost(training_inputs, training_outputs)
    _, (converged, iters, state) = adam(cost, W, W, state=state, **adam_kwargs)

    # calculate model cost
    costvalue = cost(W)
    g.message(costvalue)
    costs.append([k, k*adam_kwargs['maxiter'], costvalue])

    # save intermediate model weights
    if k >= saveweightsstart and k % saveweightsevery == 0:
        if (g.rank() == 0):
            g.save(os.path.join(snakemake.output.model_weights_dir, f"{k}"), W)

# save costs
if (g.rank() == 0):
    costs = np.array(costs)
    np.savetxt(snakemake.output.model_cost, costs)