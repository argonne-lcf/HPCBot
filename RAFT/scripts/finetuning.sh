#!/bin/bash
#PBS -l select=1
#PBS -l walltime=50:00:00
#PBS -q preemptable
#PBS -l filesystems=home
#PBS -A HPCBot
#PBS -o logs/
#PBS -e logs/
#PBS -M trungvo.usth@gmail.com

module use /soft/modulefiles
source /home/btrungvo/venv_autotrain/bin/activat5
autotrain --config /home/btrungvo/workspace/HPCBot/configs/llama3-1-70b-sft-500-2.yml