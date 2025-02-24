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
from qcdml_evals import qcdml_eigvals

# load parameters
pv = snakemake.wildcards.pv
ptctype = snakemake.wildcards.ptctype
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
n_layers = int(snakemake.wildcards.n_layers)

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op

# import specific PTC model
if ptctype == "":
    from qcdml_layer_ptc_5d import v_PTC_5d as PTC_5d
elif ptctype == "s":
    from qcdml_layer_ptc_5d import v_sPTC_5d as PTC_5d

with torch.no_grad():
    # load gauge field
    loadpath = os.path.join(gconfig_dir, gconfig)
    U = g.load(loadpath)
    L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
    grid5 = g.grid(L5, g.double)
    U5_qcdml = gptU4d_2_qcdmlU5d(U, grid5)
    
    # domain-wall operator or pauli-villars preconditioned domain wall operator
    D = qcdml_op(U, mobius_p)
    
    # ptc1hxl model from previous training
    paths = [[]] + [[(mu, 1)] for mu in range(5)] + [[(mu, -1)] for mu in range(5)]
    model = torch.nn.Sequential(*[PTC_5d(1, 1, paths, U5_qcdml) for _ in range(n_layers)])
    model.load_state_dict(torch.load(snakemake.input.model_weights))
    prec = lambda x: model.forward(torch.stack([x]))[0]
    
    # preconditioned operator
    prec_D_dw = lambda x: prec(D(x))
    
    # compute and save eigenvalues
    evals = qcdml_eigvals(L5, prec_D_dw)
    np.savetxt(snakemake.output.evals_ptc, evals)
    