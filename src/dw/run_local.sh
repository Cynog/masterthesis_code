#!/bin/bash


# run the snakemake pipeline locally
snakemake --cores all --config dimension=$1 -r ${@:2}
