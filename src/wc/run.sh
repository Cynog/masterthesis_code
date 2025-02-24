#!/bin/bash


# create folder for slurm logs
mkdir -p ../../_logs/dw

# run the snakemake pipeline on slurm cluster
snakemake --profile _profile --config dimension=$1 -r ${@:2}
