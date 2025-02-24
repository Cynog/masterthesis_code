# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
pv = snakemake.wildcards.pv
approx = snakemake.wildcards.approx
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass
Ls = snakemake.wildcards.Ls

# load evals without preconditioning
evals = np.loadtxt(snakemake.input.evals, dtype=np.complex64)

# plot
plt.scatter(0, 0, marker="+", c='black')
plt.scatter(evals.real, evals.imag, s=1, label='no prec')
approx_text = "approximate " if approx else ""
op = r"$D_\textrm{dw}$"
if pv:
    op = r"$D_\textrm{pv}^\dag D_\textrm{dw}$"
plt.title(f"{approx_text} eigenvalues of {op}\n{pv}{gconfig}_{Ls} m={mass}")
plt.xlabel('re')
plt.ylabel('im')
if pv and gconfig.startswith("4c4"):
    plt.xlim(0, 5)
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
