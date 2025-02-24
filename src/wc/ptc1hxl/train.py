# import required packages
import os
import numpy as np
import gpt as g
from gpt.qcd.fermion import wilson_clover

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_adam import adam, ADAM_State
from gpt_models import get_ptc1hxl

# load parameters
seed = snakemake.params.seed
fermion_p = snakemake.params.fermion_p
fermion_p["mass"] = float(snakemake.wildcards.mass)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
Wscale = snakemake.params.Wscale
adam_kwargs = snakemake.params.adam_kwargs
alpha_halfevery = snakemake.params.alpha_halfevery
n_layers = int(snakemake.params.n_layers)
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

# ptc1hxl model with weights for noisy identity
model = get_ptc1hxl(U, n_layers)
W = model.random_weights(rng)
for w in W:
    w *= Wscale
    w[0, :, :] += g.matrix_spin([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 4)

# train model
costs = []
state = None
for k in range(1, ntrainvectors + 1):
    # generate training data
    source = g.random.cnormal(rng, src)
    training_outputs = [source]
    training_inputs = [g(D_wc * s) for s in training_outputs] 
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
    
    # half learning rate
    if alpha_halfevery is not None and k % alpha_halfevery == 0:
        adam_kwargs['alpha'] /= 2

# save costs
if (g.rank() == 0):
    costs = np.array(costs)
    np.savetxt(snakemake.output.model_cost, costs)
