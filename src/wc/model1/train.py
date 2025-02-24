# import required packages
import os
import numpy as np
import gpt as g
from gpt.qcd.fermion import wilson_clover
from gpt.algorithms import inverter
from gpt.qcd.fermion import preconditioner

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_adam import adam, ADAM_State
from gpt_models import get_model1

# load parameters
seed = snakemake.params.seed
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
adam_kwargs = snakemake.params.adam_kwargs
solver_kwargs = snakemake.params.solver_kwargs
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

# use cg with shur complement preconditioner as inverter for training
cg = inverter.cg(solver_kwargs)
prec = preconditioner.eo2_ne(parity=g.odd)
slv = inverter.preconditioned(prec, cg)
D_wc_inv = slv(D_wc)

# model1 with multigrid setup and smoother from previous training
mg_setup_2lvl = g.load(snakemake.input.multigrid_setup)
model = get_model1(U, mg_setup_2lvl)
W_coarse_lptc = g.load(snakemake.input.coarse_lptc_weights)
W_smoother = g.load(snakemake.input.smoother_weights)
W = W_coarse_lptc + W_smoother
if (g.rank() == 0):
    g.save(os.path.join(snakemake.output.model_weights_dir, f"0"), W)

# train model
state = None
costs = []
for k in range(1, ntrainvectors + 1):
    # generate training data
    source1 = g.random.cnormal(rng, src)
    source2 = g.random.cnormal(rng, src)
    bh = g(D_wc * source1)
    uh = source1
    bl = source2
    ul = g(D_wc_inv * source2)
    training_outputs = [uh, ul]
    training_inputs = [bh, bl] 
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