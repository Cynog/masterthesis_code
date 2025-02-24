# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass

# load evals for the coarse wilson-clover operator
evals_coarse = np.loadtxt(snakemake.input.evals_coarse, dtype=np.complex64)

# plot
plt.scatter(0, 0, c='black')
plt.scatter(evals_coarse.real, evals_coarse.imag, s=1)
plt.title(f"eigenvalues of $D_\\textrm{{wc}}$ on the coarse grid\n{gconfig} m={mass}")
plt.xlabel('re')
plt.ylabel('im')
plt.savefig(snakemake.output.plot)
plt.close()
