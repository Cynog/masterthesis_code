# import required packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# configure matplotlib
mpl.rcParams['text.usetex'] = True

# load parameters
gconfig = snakemake.wildcards.gconfig
mass = snakemake.wildcards.mass
model = snakemake.wildcards.model

# load cost data
costdata = np.loadtxt(snakemake.input.cost)
cost = costdata[:, 2]

# plot cost data
plt.plot(cost)
plt.title(f"{model} cost vs training step\n{gconfig} m={mass}")
plt.xlabel("training step")
plt.ylabel("cost")
if not "model1" in model:
    plt.yscale("log")
plt.savefig(snakemake.output.plot)
plt.close()