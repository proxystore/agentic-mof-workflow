#!/bin/bash

python -m mofa.agentic.run \
      --node-path input-files/zn-paddle-pillar/node.json \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --maximum-train-size 2048 \
      --maximum-strain 0.25 \
      --retrain-freq 64 \
      --num-epochs 128 \
      --num-samples 1024 \
      --gen-batch-size 128 \
      --simulation-budget 32768 \
      --md-timesteps 1000000 \
      --md-snapshots 10 \
      --retain-lammps \
      --raspa-timesteps 50000 \
      --dft-opt-steps 1 \
      --compute-config federated \
      --cpu-endpoint 166a9195-07cc-496b-96fd-8a4be5aec97a \
      --polaris-endpoint acb10b91-e811-4d03-a0fc-dc3fb23bca0d \
      --log-level INFO
      # --lammps-on-ramdisk \
