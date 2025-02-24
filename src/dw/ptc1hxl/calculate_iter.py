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

# load parameters
pv = snakemake.wildcards.pv
ptctype = snakemake.wildcards.ptctype
seed_solve = snakemake.params.seed_solve
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
n_calciter = snakemake.params.n_calciter
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
gmres_kwargs = snakemake.params.gmres_kwargs
n_layers = int(snakemake.params.n_layers)

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op
    from qcdml_operators import qcdml_pv

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
    
    # initialize random number generator
    torch.manual_seed(seed_solve)

    # solve for n_calciter times
    iterations = []
    histories = []
    for _ in range(n_calciter):
        # random normal right hand side
        src = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
        if pv:
            src = qcdml_pv(U, mobius_p)(src)

        # solve with preconditioning
        x, info = qcd_ml.util.solver.GMRES(D, src, torch.clone(src), preconditioner=prec, **gmres_kwargs)
        print(f"{info['k']}, residual: {info['res']}")

        # save number of iterations and history
        histories.append(info['history'])
        if info["k"] == gmres_kwargs['maxiter']:
            iterations.append(-gmres_kwargs['maxiter'])
            break
        iterations.append(info['k'])

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