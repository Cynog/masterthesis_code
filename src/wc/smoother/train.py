# import required packages
import os
import numpy as np
import gpt as g
from gpt.qcd.fermion import wilson_clover

# import local librariers
import sys
sys.path.append("../../lib")
from gpt_adam import adam, ADAM_State
from gpt_models import get_ptc1hxl, get_smoother

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

# ptc1h1l model from previous training
model_ptc1h1l = get_ptc1hxl(U, 1)
W_ptc1h1l = g.load(snakemake.input.weights_ptc1h1l)
M_ptc1h1l = model_ptc1h1l(W_ptc1h1l)

# smoother model with weights for noisy identity
model = get_smoother(U)
W = model.random_weights(rng)
for w in W:
    w *= Wscale
    w[0, :, :] += g.matrix_spin([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 4)

def iterative_uk(ukm1, b, Mh, D):
    result = g.copy(b)
    result @= result - D*ukm1
    result @= Mh * result
    result @= ukm1 + result
    return result

def ukn(b, Mh, D, n):
    uk = g.copy(b)
    uk[:] = 0
    for k in range(n):
        uk @= iterative_uk(uk, b, Mh, D)
    return uk

# train model
costs = []
state = None
for k in range(1, ntrainvectors+1):
    # generate training data
    source = g.random.cnormal(rng, src)
    # FIXME: This uses k = 0, r = 2, u_0 = 0
    training_outputs = [ukn(source, M_ptc1h1l, D_wc, 0+2)]
    # FXIME: I don't like this XXX
    training_inputs = [[source, ukn(source, M_ptc1h1l, D_wc, 0+0)] for _ in training_outputs] 
    normalizations = [g.norm2(inp[0]) for inp in training_inputs]
    training_inputs = [[g(inp[0] / norm**0.5), g(inp[1] / norm**0.5)] for inp, norm in zip(training_inputs, normalizations)]
    training_outputs = [g(oup / norm**0.5) for oup,norm in zip(training_outputs, normalizations)]
    
    # optimize
    cost = model.cost(training_inputs, training_outputs)
    _, (converged, iters, state) = adam(cost, W, W, state=state, **adam_kwargs)

    # calculate model lost
    costvalue = cost(W)
    g.message(costvalue)
    costs.append([k, k*adam_kwargs['maxiter'], costvalue])
    
    # save intermediate model weights
    if k >= saveweightsstart and k % saveweightsevery == 0:
        if(g.rank() == 0):
            g.save(os.path.join(snakemake.output.model_weights_dir, f"{k}"), W)
    
# save costs
if (g.rank() == 0):
    costs = np.array(costs)
    np.savetxt(snakemake.output.model_cost, costs)
    