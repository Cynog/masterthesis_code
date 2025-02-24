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

# load evals without preconditioning
evals = np.loadtxt(snakemake.input.evals, dtype=np.complex64)

# load evals with model1 preconditioning
evals_model1 = np.loadtxt(snakemake.input.evals_model1, dtype=np.complex64)

# plot
#plt.scatter(evals.real, evals.imag, s=1, label='no prec')
plt.scatter(evals_model1.real, evals_model1.imag, s=1, label='model1')
plt.scatter(0, 0, c='black')
plt.title(f"eigenvalues of $M D_\\textrm{{wc}}$ for different $M$\n{gconfig} m={mass} ntrain={ntrain}")
plt.xlabel('re')
plt.ylabel('im')
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
