#!/bin/bash


snakemake --profile _profile --config dimension=$1 -r ${@:2}
