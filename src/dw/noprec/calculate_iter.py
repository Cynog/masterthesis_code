# import required packages
import os
import numpy as np
import gpt as g
import qcd_ml
import torch

# import local libraries
import sys
sys.path.append("../../lib")

# load parameters
seed_solve = snakemake.params.seed_solve
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
pv = snakemake.wildcards.pv
n_calciter = snakemake.params.n_calciter
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
gmres_kwargs = snakemake.params.gmres_kwargs

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_dw as qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op
    from qcdml_operators import qcdml_pv

# load gauge field
loadpath = os.path.join(gconfig_dir, gconfig)
U = g.load(loadpath)
L5 = [mobius_p['Ls']] + U[0].grid.fdimensions

# domain-wall operator or pauli-villars preconditioned domain wall operator
D = qcdml_op(U, mobius_p)

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

    # solve without preconditioning
    x, info = qcd_ml.util.solver.GMRES(D, src, torch.clone(src), **gmres_kwargs)
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
