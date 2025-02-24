# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass

# load evals without preconditioning
evals = np.loadtxt(snakemake.input.evals, dtype=np.complex64)

# load evals with ptc_1h1l preconditioning
evals_mg = np.loadtxt(snakemake.input.evals_mg, dtype=np.complex64)

# plot
plt.scatter(evals.real, evals.imag, s=1, label='no prec')
plt.scatter(evals_mg.real, evals_mg.imag, s=1, label='multigrid')
plt.scatter(0, 0, c='black')
plt.title(f"eigenvalues of $M D_\\textrm{{wc}}$ for different $M$\n{gconfig} m={mass}")
plt.xlabel('re')
plt.ylabel('im')
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
