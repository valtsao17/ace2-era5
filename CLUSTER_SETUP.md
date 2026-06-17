# Cluster setup

## What to scp

Just this folder (`ace2_lag_cluster/`). The large files live separately — see below.

## Large files (scp these separately)

| What | Location on current machine | Size |
|---|---|---|
| Model checkpoint | `/home/jovyan/ace2-era5/checkpoint/ace2_era5_ckpt.tar` | ~3.4 GB |
| Initial conditions | `/home/jovyan/ace2-era5/data/lag_data/initial_conditions/` | ~15 GB total |
| Forcing files | `/home/jovyan/ace2-era5/data/lag_data/forcing_data_ace2era5/` | included in above |

Put them wherever you want on the cluster and pass the paths via flags (see below).

## Install

```bash
pip install fme pyyaml xarray netcdf4
```

## Run (4 GPUs)

```bash
python scripts/run_lag_inference_nov.py \
  --root /path/to/your/home \
  --checkpoint /path/to/ace2_era5_ckpt.tar \
  --ic-dir /path/to/initial_conditions \
  --forcing-dir /path/to/forcing_data_ace2era5 \
  --output-dir /path/to/outputs/runs \
  --years 1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000 \
  --n-gpus 4 \
  --skip-existing \
  --forward-steps-in-memory 40
```

If your paths follow the default layout (`<root>/checkpoint/`, `<root>/data/lag_data/`, etc.) you can just pass `--root` and skip the individual path flags.

## How the 4-GPU parallelism works

Members are dispatched round-robin across GPUs (member 0 → GPU 0, member 1 → GPU 1, ..., member 4 → GPU 0, ...). Each member is an independent subprocess with `CUDA_VISIBLE_DEVICES` set to its assigned GPU. With 25 members and 4 GPUs, you'll have 4 running at once at all times until the last batch.

## Skip existing / resume

`--skip-existing` validates each member's output netCDF before skipping — checks that `TMP2m` is present and has 500 timesteps. Safe to use when resuming a crashed run.
