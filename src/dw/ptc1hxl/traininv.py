# import required packages
import os
import numpy as np
import gpt as g
import qcd_ml
import torch

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_convert import gptU4d_2_qcdmlU5d


def l2norm(v):
    """Calculate the l2-norm of a torch tensor"""
    return torch.sqrt((v * v.conj()).real.sum())


def l2err_squared(v1, v2):
    """Calculate the squared l2-error between two torch tensors"""
    err = (v2 - v1)
    return (err * err.conj()).real.sum()


# load parameters
pv = snakemake.wildcards.pv
ptctype = snakemake.wildcards.ptctype
seed = snakemake.params.seed
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
Wscale = snakemake.params.Wscale
adam_kwargs = snakemake.params.adam_kwargs
solver_kwargs = snakemake.params.solver_kwargs
n_layers = int(snakemake.params.n_layers)
ntrainvectors = snakemake.params.ntrainvectors
saveweightsstart = snakemake.params.saveweightsstart
saveweightsevery = snakemake.params.saveweightsevery
alpha_halfevery = snakemake.params.alpha_halfevery

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
    from qcdml_operators import qcdml_dw_inv as qcdml_op_inv
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op
    from qcdml_operators import qcdml_pv_dw_inv as qcdml_op_inv

# import specific PTC model
if ptctype == "":
    from qcdml_layer_ptc_5d import v_PTC_5d as PTC_5d
elif ptctype == "s":
    from qcdml_layer_ptc_5d import v_sPTC_5d as PTC_5d

# initialize random number generator
torch.manual_seed(seed)

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
grid5 = g.grid(L5, g.double)
U5_qcdml = gptU4d_2_qcdmlU5d(U, grid5)

# domain wall operator or pauli-villars preconditioned domain wall operator
D = qcdml_op(U, mobius_p)

# inverse domain-wall operator or inverse pauli-villars preconditioned domain wall operator
D_inv = qcdml_op_inv(U, mobius_p, solver_kwargs)

# ptc1hxl model with weights initialized to noisy identity
paths = [[]] + [[(mu, 1)] for mu in range(5)] + [[(mu, -1)] for mu in range(5)]
model = torch.nn.Sequential(*[PTC_5d(1, 1, paths, U5_qcdml, Wscale=Wscale) for _ in range(n_layers)])

# save starting model weights
torch.save(model.state_dict(), os.path.join(snakemake.output.model_weights_dir, "0"))

# get cost scaling factor
if ptctype == "":
    cost_scale = np.prod(L5)
elif ptctype == "s":
    cost_scale = np.prod(L5[1:])

# train model
optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
loss = np.zeros(ntrainvectors)
for k in range(1, ntrainvectors + 1):
    # generate training data
    src1 = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
    Dsrc1 = D(src1)
    nrm1 = l2norm(Dsrc1)
    inp1 = torch.stack([Dsrc1 / nrm1])
    out1 = torch.stack([src1 / nrm1])

    src2 = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
    Dinvsrc2 = D_inv(src2)
    nrm2 = l2norm(src2)
    inp2 = torch.stack([src2 / nrm2])
    out2 = torch.stack([Dinvsrc2 / nrm2])

    # calculate cost
    cost1 = l2err_squared(model.forward(inp1), out1)
    cost2 = l2err_squared(model.forward(inp2), out2)
    cost = cost1 + cost2
    cost /= cost_scale

    # optimize
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    loss[k - 1] = cost.item()

    # save intermediate model weights
    if k >= saveweightsstart and k % saveweightsevery == 0:
        print(f"{k:4d} ({k / ntrainvectors * 100: 3.2f} %): {loss[k-1]:.2e}")
        torch.save(model.state_dict(), os.path.join(snakemake.output.model_weights_dir, f"{k}"))

# save costs
np.savetxt(snakemake.output.model_cost, loss)
