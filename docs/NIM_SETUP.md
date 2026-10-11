# Running NVIDIA BioNeMo NIMs locally

A NIM is a GPU container that serves its model over HTTP. This console speaks that
HTTP API, so a NIM container is the only thing standing between the app and real
predictions.

## 1. Host requirements

| Requirement | Notes |
| --- | --- |
| NVIDIA GPU | Ampere or newer for most BioNeMo NIMs; check the model's profile for VRAM (ESMFold ≈ 8 GB, DiffDock ≈ 8 GB, Evo 2 40B is very large) |
| Driver | Recent datacenter or studio driver (`nvidia-smi` must work) |
| Docker (or Podman) | With the **nvidia-container-toolkit** so containers see the GPU |
| NGC API key | Free at <https://build.nvidia.com> → *Get API Key* |

Verify the GPU is visible to containers before anything else:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that prints a GPU table, you are ready. If it fails, install
`nvidia-container-toolkit` and restart Docker — nothing else in this document will
work until it does.

## 2. Keys and login

```bash
export NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxx

# nvcr.io requires the literal username $oauthtoken
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The console reads `NGC_API_KEY` (or `NVIDIA_API_KEY`) itself: it enables NVIDIA's
hosted endpoints and the optional language-model provider. It never writes the key
to disk.

## 3. Start a NIM

Easiest path — the helper script starts a sensible default set and waits for
readiness:

```bash
bash scripts/start_nims.sh                 # esmfold, molmim, diffdock, proteinmpnn
bash scripts/start_nims.sh esmfold         # or just one
```

It prints a registration URL per model. The equivalent manual command is:

```bash
docker run -d --rm --name nim-esmfold \
  --gpus all --shm-size=16g \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_CACHE_PATH=/opt/nim/.cache \
  -u $(id -u) \
  -v $HOME/.cache/nim/esmfold:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/meta/esmfold:latest
```

| Model | Image | VRAM (approx) |
| --- | --- | --- |
| ESMFold | `nvcr.io/nim/meta/esmfold:latest` | 8 GB |
| AlphaFold2 | `nvcr.io/nim/deepmind/alphafold2:latest` | 16 GB + MSA databases (~1 TB for full accuracy) |
| OpenFold3 | `nvcr.io/nim/nvidia/openfold3:latest` | 16 GB |
| Boltz-2 | `nvcr.io/nim/nvidia/boltz2:latest` | 16 GB |
| MolMIM | `nvcr.io/nim/nvidia/molmim:latest` | 8 GB |
| GenMol | `nvcr.io/nim/nvidia/genmol:latest` | 8 GB |
| DiffDock | `nvcr.io/nim/mit/diffdock:latest` | 8 GB |
| ProteinMPNN | `nvcr.io/nim/ipd/proteinmpnn:latest` | 4 GB |
| RFdiffusion | `nvcr.io/nim/ipd/rfdiffusion:latest` | 8 GB |
| ESM-2 | `nvcr.io/nim/meta/esm2-650m:latest` | 8 GB |
| Evo 2 | `nvcr.io/nim/arc/evo2-40b:latest` | 80 GB (multi-GPU profiles exist) |

First start downloads several GB into the cache directory you mounted; later
starts take seconds.

## 4. Verify and register

```bash
curl -s http://localhost:8000/v1/health/ready     # {"status":"ready"}
curl -s http://localhost:8000/v1/models           # model metadata
curl -s http://localhost:8000/docs | head         # OpenAPI, if exposed
```

In the console: **Administration → Backends**, set each model's base URL (the
defaults are already `http://localhost:8000`), press *Probe every configured
endpoint*, then run something. The result badge reads `LOCAL NIM`.

Multiple NIMs run side by side on different ports (`-p 8001:8000`, …); the console
stores one endpoint per model, so mixing ports is normal.

### Docker Compose

`docker-compose.yml` ships a `console` service plus one NIM behind a `gpu`
profile:

```bash
NGC_API_KEY=nvapi-xxx docker compose --profile gpu up --build
```

Inside a compose network, refer to a NIM by its service name
(`http://nim-esmfold:8000`), not `localhost`.

## 5. Paths change between releases

NVIDIA moves NIM endpoints between versions (for example DiffDock has been served
under both `/v1/biology/mit/diffdock` and `/molecular-docking/diffdock/generate`).
The console therefore:

1. tries the path recorded in its catalogue,
2. on 404 asks `/docs`, `/v1/models` and `/v1/openapi.json` for the real path and
   caches what it learns,
3. reports the outcome — including the exact URL it used — in the run's
   provenance, with a `curl` command you can paste yourself.

If a model misbehaves, the `path_confidence` marker in `nims/catalog.py` tells you
how well documented that entry is; `verified` means it was confirmed against a
live service, `configurable` means the path is a setting you may need to adjust.

## 6. Hosted NIMs (no GPU needed)

With an NGC key set, the console can call NVIDIA's hosted endpoints at
`integrate.api.nvidia.com`. Choose **NVIDIA hosted NIM only** as the backend, or
leave it on `Auto` to prefer a local container and fall back to hosted. This is
rate-limited and not suitable for bulk screening, but excellent for trying
individual predictions.

## 7. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `docker: Error response from daemon: could not select device driver` | nvidia-container-toolkit missing or Docker not restarted after installing it |
| `unauthorized: authentication required` | Not logged in to `nvcr.io`, or the username was not the literal `$oauthtoken` |
| Container starts then exits immediately | Read the logs: `docker logs nim-esmfold`. Usually VRAM or a missing cache directory permission |
| Console still says `SIMULATED` | The endpoint for that model is empty or wrong, or the container is not ready — use *Probe every configured endpoint* |
| `CUDA out of memory` | A smaller NIM, a smaller batch, or a bigger GPU; `--shm-size=16g` also helps for some models |
| Runs take minutes on first call | Models load weights lazily on the first request; subsequent calls are fast |
| Wrong results from a renamed path | Set the path override in the catalogue, or rely on discovery and check the provenance URL |
