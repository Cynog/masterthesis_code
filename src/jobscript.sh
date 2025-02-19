#!/bin/bash


# pre-job commands
echo "Running on cluster $CLUSTER"
if [[ "$CLUSTER" == "qp4" ]] 
then
    module load gpt/2023-12-11
    export PYTHONPATH=$PYTHONPATH:$HOME/.venv_qp4/lib/python3.9/site-packages/
elif [[ "$CLUSTER" == "hpd" ]]
then
    module use $HOME/hpd/site/modulefiles
    module load gpt-hpd
fi

# run snakemake job
{exec_job}
