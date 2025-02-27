# masterthesis_code

This repository contains code to train and evaluate the machine-learning models discussed in my master's thesis.

The src folder is split into two subfolders. `./src/wc` includes all code specific to the 4-dimensional Wilson-clover Dirac operator and is based on the [Grid Python Toolkit](https://github.com/lehner/gpt) library. `./src/dw` includes all code specific to the 5-dimensional Möbius Domain-wall Dirac operator. It uses wrappers for the Domain-wall operator from GPT, but is based on the [qcd_ml](https://github.com/daknuett/qcd_ml) library.

[Snakemake](https://snakemake.readthedocs.io/en/stable/) was used as a workflow manager to run the jobs on the high-performance-computing clusters qp4 and hpd of the University of Regensburg.

There is not the single one ideal workflow manager. One problem with Snakemake I encountered was, that a rules output may not depend on a variable that depends on a wildcard. Let's say for example, we want to learn a model with two different cost functions. I did not find a way to write a single rule with these cost functions as wildcards in the case, that a different number of training steps is used for both cost functions and that the output depends on the number of training steps. Thats why some rules seem redundant and I require to specify the configs dimension on the command line to be able to use for example a different learning rate depending on the lattice size. All in all, Snakemake still proved to be very useful for me.

## Installation

The code is intended to be run on the [slurm](https://slurm.schedmd.com/documentation.html) computing clusters qp4 and hpd of the University of Regensburg. Local execution is possible, but not recommended, either by installing gpt and qcd_ml locally or by running the docker [devcontainer](https://containers.dev/) included in this repository.

The following files are specific to the HPC clusters at the University of Regensburg and might have to be changed for running on other clusters:\
`./src/jobscript.sh`\
`./src/*/_profile/config.yaml`\
`./src/*/run.sh` (should work with other slurm clusters, as it just references the configures profiles)\
`./src/*/Snakefile` (only specified resources might have to be changed)

## Execution

To generate all results for the Wilson-clover operator, run the following from the `./src/wc` folder

```bash
./run.sh 4c8 full
./run.sh 8c16 full
```

To generate all results for the Domain-wall operator, run the following from the `./src/dw` folder
```bash
./run.sh 4c4 full pv_full
./run.sh 8c16 full pv_full
```

Of course, also more specific rules can be requested like
```bash
./run.sh 4c4 noprec_full
```
or even single output files like
```bash
./run.sh 4c8 ../../_output/wc/4c8_0600.cfg/iter/-0.590.txt
```

### Local Execution

For local execution, run the rule RULE like
```bash
snakemake --cores all --jobs 1 --config dimension=DIM -r RULE
```
with all available cores and one job at a time. DIM is either 4c8 or 8c16 for the Wilson-clover operator and 4c4 or 8c16 for the Domain-wall operator.

## Acknowledgements

According the implementation accompanying my master's thesis, my gratitude belongs to [@daknuett](https://github.com/daknuett) and [@simon-pfahler](https://github.com/simon-pfahler). They supported me throughout the whole process, @daknuett provided me with code in the beginning to get me started and both of them worked together to create the [qcd_ml](https://github.com/daknuett/qcd_ml) library. It is well documented and nice to use. Modifying the code to work for the 5-dimensional Möbius Domain-wall operator was straightforward.
