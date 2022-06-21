#!/bin/bash
#SBATCH --job-name=gpt-j-run
#SBATCH --partition short
#SBATCH --mem 60000
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=00:30:00
poetry run python -m ltp run-cmv 30522_gpt_j_test
