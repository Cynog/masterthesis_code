# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass
ntrain = snakemake.wildcards.ntrain
n_layers = snakemake.wildcards.n_layers
costf = snakemake.wildcards.costf

# load evals without preconditioning
evals = np.loadtxt(snakemake.input.evals, dtype=np.complex64)

# load evals with ptc_1h1l preconditioning
evals_ptc = np.loadtxt(snakemake.input.evals_ptc, dtype=np.complex64)

# plot
#plt.scatter(evals.real, evals.imag, s=1, label='no prec')
plt.scatter(evals_ptc.real, evals_ptc.imag, s=1, label=f"ptc1h{n_layers}l{costf} prec")
plt.scatter(0, 0, c="black")
plt.title(f"eigenvalues of $M D_\\textrm{{wc}}$ for different $M$\n{gconfig} m={mass} ntrain={ntrain}")
plt.xlabel("re")
plt.ylabel("im")
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
