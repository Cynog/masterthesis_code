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
n_layers = snakemake.params.n_layers
ntrainvectors = snakemake.params.ntrainvectors
saveweightsstart = snakemake.params.saveweightsstart
saveweightsevery = snakemake.params.saveweightsevery
Ls = snakemake.wildcards.Ls
ptctype = snakemake.params.ptctype
costf = snakemake.params.costf

# ntrain values
ntrain_plot_base = [0] + list(range(saveweightsstart, ntrainvectors+1, saveweightsevery))

# load iterations without preconditioning
iter_noprec = np.loadtxt(snakemake.input.iter, dtype=int)

# check if gain calculation is possible
gain = True
if np.min(iter_noprec) < 0:
    gain = False

if "" in ptctype:
    # load iterations with ptc1hxl preconditioning
    iter_ptc = []
    for smi in snakemake.input.iter_ptc1hxl:
        iter_ptc.append(np.loadtxt(smi, dtype=int))

    # filter for solved cases and calculate average and error
    ntrain_plot_ptc = [ntrain_plot_base[i] for i in range(len(ntrain_plot_base)) if np.min(iter_ptc[i]) > 0]
    iter_ptc = [ip for ip in iter_ptc if np.min(ip) > 0]
    if gain:
        iter_ptc = [iter_noprec / ip for ip in iter_ptc]
    iter_ptc_avg = [np.mean(ip) for ip in iter_ptc]
    iter_ptc_err = [np.std(ip) / np.sqrt(len(ip)) for ip in iter_ptc]

if "s" in ptctype:
    # load iterations with sptc1hxl preconditioning
    iter_sptc = []
    for smi in snakemake.input.iter_sptc1hxl:
        iter_sptc.append(np.loadtxt(smi, dtype=int))

    # filter for solved cases and calculate average and error
    ntrain_plot_sptc = [ntrain_plot_base[i] for i in range(len(ntrain_plot_base)) if np.min(iter_sptc[i]) > 0]
    iter_sptc = [ip for ip in iter_sptc if np.min(ip) > 0]
    if gain:
        iter_sptc = [iter_noprec / ip for ip in iter_sptc]
    iter_sptc_avg = [np.mean(ip) for ip in iter_sptc]
    iter_sptc_err = [np.std(ip) / np.sqrt(len(ip)) for ip in iter_sptc]

# plot
if "" in ptctype:
    plt.errorbar(ntrain_plot_ptc, iter_ptc_avg, yerr=iter_ptc_err, fmt='o-', label=f"ptc1h{n_layers}l{costf}")
if "s" in ptctype:
    plt.errorbar(ntrain_plot_sptc, iter_sptc_avg, yerr=iter_sptc_err, fmt='o-', label=f"sptc1h{n_layers}l{costf}")
if gain:
    plt.title(f"ptc1h{n_layers}l{costf} iteration count gain vs training step\ngconfig={pv}{gconfig}_{Ls} mass={mass}")
    plt.ylabel('iteration count gain')
else:
    plt.title(f"ptc1h{n_layers}l{costf} iteration count vs training step\n{pv}{gconfig}_{Ls} mass={mass}")
    plt.ylabel('iteration count')
plt.xlabel('training step')
plt.xlim(0, ntrainvectors)
plt.ylim(bottom=1)
plt.legend()
plt.savefig(snakemake.output.plot)
plt.close()
