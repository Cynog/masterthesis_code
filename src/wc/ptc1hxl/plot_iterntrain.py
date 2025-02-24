# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass
n_layers = snakemake.params.n_layers
costf = snakemake.params.costf
ntrainvectors = snakemake.params.ntrainvectors
saveweightsstart = snakemake.params.saveweightsstart
saveweightsevery = snakemake.params.saveweightsevery

# ntrain values
ntrain_plot = list(range(saveweightsstart, ntrainvectors + 1, saveweightsevery))

# load iterations without preconditioning
iter_noprec = np.loadtxt(snakemake.input.iter, dtype=int)

# check if gain calculation is possible
gain = True
if np.min(iter_noprec) < 0:
    gain = False

# load iterations with model1 preconditioning
iter_plot = []
for smi in snakemake.input.iter_ptc1hxl:
    iter_plot.append(np.loadtxt(smi, dtype=int))

# filter for solved cases
ntrain_plot = [ntrain_plot[i] for i in range(len(ntrain_plot)) if np.min(iter_plot[i]) > 0]
iter_plot = [ip for ip in iter_plot if np.min(ip) > 0]
if gain:
    iter_plot = [iter_noprec / ip for ip in iter_plot]
iter_plot_avg = [np.mean(ip) for ip in iter_plot]
iter_plot_err = [np.std(ip) / np.sqrt(len(ip)) for ip in iter_plot]

# plot
plt.errorbar(ntrain_plot, iter_plot_avg, yerr=iter_plot_err, fmt='o-')
if gain:
    plt.title(f"ptc1h{n_layers}l{costf} iteration count gain vs training step\ngconfig={gconfig} mass={mass}")
    plt.ylabel("iteration count gain")
else:
    plt.title(f"ptc1h{n_layers}l{costf} iteration count vs training step\ngconfig={gconfig} mass={mass}")
    plt.ylabel("iteration count")
plt.xlabel("training step")
plt.xlim(0, ntrainvectors)
plt.ylim(bottom=1)
plt.savefig(snakemake.output.plot)
plt.close()
