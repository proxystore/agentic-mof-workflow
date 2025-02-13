#!/bin/bash -le
#PBS -l select=6:system=polaris
#PBS -l walltime=0:60:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug-scaling
#PBS -N mofa-test
#PBS -A examol

hostname
pwd

# Activate the environment
# conda activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris
# which python

# Launch MPS on each node
# NNODES=`wc -l < $PBS_NODEFILE`
 #mpiexec -n ${NNODES} --ppn 1 ./bin/enable_mps_polaris.sh &

python -m mofa.agentic.run \
      --node-path input-files/zn-paddle-pillar/node.json \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --maximum-train-size 8192 \
      --maximum-strain 0.5 \
      --retrain-freq 1 \
      --num-epochs 4 \
      --num-samples 128 \
      --gen-batch-size 64 \
      --simulation-budget 8 \
      --md-timesteps 100000 \
      --md-snapshots 10 \
      --raspa-timesteps 10000 \
      --lammps-on-ramdisk \
      --dft-opt-steps 2 \
      --compute-config polaris-single \
      --ray-address localhost:6379
echo Python done

# Shutdown services
# mpiexec -n ${NNODES} --ppn 1 ./bin/disable_mps_polaris.sh
