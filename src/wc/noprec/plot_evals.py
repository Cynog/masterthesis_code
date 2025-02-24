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

# plot
plt.scatter(evals.real, evals.imag, s=1)
plt.scatter(0, 0, c='black')
plt.title(f"eigenvalues of $D_\\textrm{{wc}}$\n{gconfig} m={mass}")
plt.xlabel('re')
plt.ylabel('im')
plt.savefig(snakemake.output.plot)
plt.close()
