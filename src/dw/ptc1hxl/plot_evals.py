# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
pv = snakemake.wildcards.pv
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass
Ls = snakemake.wildcards.Ls
ntrain = snakemake.wildcards.ntrain
n_layers = snakemake.wildcards.n_layers
costf = snakemake.params.costf
ptctype = snakemake.params.ptctype

# load evals without preconditioning
evals = np.loadtxt(snakemake.input.evals, dtype=np.complex64)

# load evals with preconditioning
if "" in ptctype:
    evals_ptc = np.loadtxt(snakemake.input.evals_ptc, dtype=np.complex64)
if "s" in ptctype:
    evals_sptc = np.loadtxt(snakemake.input.evals_sptc, dtype=np.complex64)

# plot
plt.scatter(0, 0, marker="+", c='black')
#plt.scatter(evals.real, evals.imag, s=1, label='no prec')
if "" in ptctype:
    plt.scatter(evals_ptc.real, evals_ptc.imag, s=1, alpha=0.5, label=f"ptc_1h{n_layers}l{costf}")
if "s" in ptctype:
    plt.scatter(evals_sptc.real, evals_sptc.imag, s=1, alpha=0.5, label=f"sptc_1h{n_layers}l{costf}")
op = r"$M D_\textrm{dw}$"
if pv:
    op = r"$M D_\textrm{pv}^\dag D_\textrm{dw}$"
plt.title(f"eigenvalues of {op} for different $M$\n{pv}{gconfig}_{Ls} m={mass} ntrain={ntrain}")
plt.xlabel('re')
plt.ylabel('im')
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
