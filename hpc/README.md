# Running on a SLURM cluster

One-time setup, then every run is a single `sbatch` command. All heavy
files (conda env, HF model cache) live on `/scratch/users/$USER`, not
in the quota-limited home directory.

## 1. Get the code onto the cluster

Either clone from a Git host, or copy from the local machine:

```powershell
scp -r <local-repo-path> <user>@<hpc-host>:/scratch/users/<user>/
```

(Exclude `.venv/` and any large `results/` from the copy, or clone
fresh instead.)

## 2. One-time setup (login node)

```bash
cd /scratch/users/$USER/<repo>
bash hpc/setup_env.sh
```

This creates a Python 3.11 conda env at `/scratch/users/$USER/envs/t2t`,
installs `t2t` with the GPU extra (bitsandbytes), runs the unit tests,
and pre-downloads Qwen2.5-7B-Instruct and the NLI model so jobs can run
with `HF_HUB_OFFLINE=1` (compute nodes often have no internet access).

## 3. Submit a run

```bash
sbatch --time=00:30:00 hpc/submit_run.sh configs/e2e_smoke_cpu.yaml                     # smoke test
sbatch hpc/submit_run.sh   configs/battery_full/rotowire_faithful.yaml                  # generation + evaluation
sbatch hpc/submit_train.sh configs/arm2_rotowire_hpc_aug_filtered.yaml                  # encoder training
```

One `submit_run.sh` job runs generation, evaluation
(`--claims --parent`), and a 5-item preview in the log. Generation and
training are resumable: resubmitting the same config continues where it
stopped.

Monitor and inspect:

```bash
squeue -u $USER               # queue state
tail -f hpc/logs/t2t-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

## 4. Sync results back

Results land in `results/<name>-<hash>/` exactly as on a workstation.
Copy them back for the dashboard and the report scripts:

```powershell
scp -r <user>@<hpc-host>:/scratch/users/<user>/<repo>/results/* <local-repo-path>\results\
```

## Cluster assumptions (verify once)

- GPUs are A100 MIG slices: `1g.5gb` slices (5 GB, too small for the 7B
  model but enough for NLI evaluation or 0.5B smoke tests) and
  `3g.20gb` slices (20 GB, fits Qwen2.5-7B in 4-bit). `submit_run.sh`
  requests `--gres=gpu:3g.20gb:1`. A CUDA process can only use one MIG
  slice, so never request more than one per job.
- Conda is expected at `/opt/software/uoa/apps/miniconda3` (edit
  `setup_env.sh` if the cluster differs).
- Scratch is `/scratch/users/$USER`. If the cluster purges scratch,
  keep the repo in `$HOME` and only the cache and env on scratch.
