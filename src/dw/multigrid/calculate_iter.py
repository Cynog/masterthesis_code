# import required packages
import os
import numpy as np
import gpt as g
import qcd_ml
from qcd_ml.util.solver import GMRES
import torch

# import local librariers
import sys
sys.path.append("../../lib")
from qcdml_multigrid_5d import ZPP_Multigrid_5d


# wrapper for counting operator applications
class counting:
    def __init__(self, Q):
        self.Q = Q
        self.k = 0
    def __call__(self, x):
        self.k += 1
        return self.Q(x)


# load parameters
pv = snakemake.wildcards.pv
seed_solve = snakemake.params.seed_solve
mobius_p = snakemake.params.mobius_p
mobius_p["mass"] = float(snakemake.wildcards.mass)
mobius_p["Ls"] = int(snakemake.wildcards.Ls)
n_calciter = snakemake.params.n_calciter
gconfig_dir = snakemake.params.gconfig_dir
gconfig = snakemake.wildcards.gconfig
gmres_kwargs = snakemake.params.gmres_kwargs
inner_solver_kwargs = snakemake.params.inner_solver_kwargs
smoother_kwargs = snakemake.params.smoother_kwargs

# import domain-wall operator or pauli-villars preconditioned domain wall operator
if pv == "":
    from qcdml_operators import qcdml_op
else:
    from qcdml_operators import qcdml_pv_dw as qcdml_op
    from qcdml_operators import qcdml_pv

with torch.no_grad():
    # load gauge field
    loadpath = os.path.join(gconfig_dir, gconfig)
    U = g.load(loadpath)
    L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
    
    # domain-wall operator or pauli-villars preconditioned domain wall operator
    D = qcdml_op(U, mobius_p)
    
    # load multigrid setup
    mg_setup = ZPP_Multigrid_5d.load(snakemake.input.multigrid_setup)
    class Lvl2MultigridPreconditioner:    
        def __init__(self, q, mg_setup, q_coarse, inner_solver_kwargs, smoother_kwargs):    
            self.q = q 
            self.mg_setup = mg_setup
            self.q_coarse = q_coarse
            self.inner_solver_kwargs = inner_solver_kwargs    
            self.smoother_kwargs = smoother_kwargs    

        def __call__(self, b):    
            x_coarse, info_coarse = GMRES(self.q_coarse, self.mg_setup.v_project(b), self.mg_setup.v_project(b), **self.inner_solver_kwargs)    
            x = self.mg_setup.v_prolong(x_coarse)
            x, info_smoother = qcd_ml.util.solver.GMRES(self.q, torch.clone(b), torch.clone(x), **self.smoother_kwargs)    
            return x
    counting_D = counting(D)
    counting_D_coarse = counting(mg_setup.get_coarse_operator(D))
    prec = Lvl2MultigridPreconditioner(counting_D, mg_setup, counting_D_coarse, inner_solver_kwargs, smoother_kwargs)
    
    # initialize random number generator
    torch.manual_seed(seed_solve)

    # solve for n_calciter times
    iterations = []
    histories = []
    opcounts = []
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
        opcounts.append((counting_D.k, counting_D_coarse.k))
        counting_D.k = 0
        counting_D_coarse.k = 0
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
    
    # save operator counts
    with open(snakemake.output.opcounts, "w") as fout:
        for opc, opc_coarse in opcounts:
            fout.write(f"{opc} {opc_coarse}\n")
