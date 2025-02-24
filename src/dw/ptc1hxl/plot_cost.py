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
n_layers = snakemake.wildcards.n_layers
costf = snakemake.wildcards.costf
ptctype = snakemake.params.ptctype

# load cost data
if "" in ptctype:
    cost_ptc = np.loadtxt(snakemake.input.cost_ptc)
if "s" in ptctype:
    cost_sptc = np.loadtxt(snakemake.input.cost_sptc)
    
# plot cost data
if "" in ptctype:
    plt.plot(cost_ptc, label=f"ptc_1h{n_layers}l{costf}")
if "s" in ptctype:
    plt.plot(cost_sptc, label=f"sptc_1h{n_layers}l{costf}")
plt.title(f"ptc1h{n_layers}l{costf} cost vs training step\n{pv}{gconfig}_{Ls} m={mass}")
plt.xlabel("training step")
plt.ylabel("cost")
plt.yscale("log")
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()