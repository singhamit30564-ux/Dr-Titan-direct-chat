# Deployment

The console is a single Streamlit process plus a SQLite file, so deploying it means
deciding where it runs relative to the GPU services, who can reach it, and how its
state is kept.

## The three shapes

**1. Laptop, all local (default).** Console and NIM containers on the same machine.
Nothing leaves the host. This is what `scripts/run_console.sh` and
`scripts/start_nims.sh` set up.

**2. Console in front of a GPU box.** The console is CPU-only and talks HTTP, so it
can run anywhere with a route to the NIMs:

```bash
BIONEMO_NIM_URL=http://gpu-box.internal:8000 BIONEMO_BACKEND=local \
  bash scripts/run_console.sh
```

**3. Containers (compose).** `docker compose --profile gpu up --build` on the GPU
host, or the `console` service alone pointed at NIMs elsewhere.

## Container

```bash
docker build -t dr-titan-console:1.0.0 .
docker run --rm -p 8501:8501 \
  -e BIONEMO_NIM_URL=http://host.docker.internal:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$PWD/.runtime:/app/.runtime" \
  dr-titan-console:1.0.0
```

The image is CPU-only and small: it contains the console, not the models. It runs
as a non-root user, has a `HEALTHCHECK` on `/healthz`, and fails the build if the
vendored 3D viewer is missing (so a 3D pane can never silently degrade).

## Configuration reference

Everything is optional; the app starts with defaults and tells you what it lacks.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BIONEMO_BACKEND` | `auto` | `auto` \| `local` \| `hosted` \| `framework` \| `simulate` |
| `BIONEMO_NIM_URL` | `http://localhost:8000` | Default endpoint for every model without an explicit one |
| `BIONEMO_HOSTED_URL` | `https://integrate.api.nvidia.com/v1` | Hosted NIM base (for proxies/mirrors) |
| `NGC_API_KEY` / `NVIDIA_API_KEY` | — | Enables hosted NIMs, `nvcr.io` pulls, and the LLM provider |
| `BIONEMO_CONSOLE_HOME` | `./.runtime` | Where the database and artifacts live |
| `BIONEMO_LLM_PROVIDER` | `auto` | `off` \| `auto` \| `local` \| `ollama` \| `nvidia` \| `openai` |
| `LOCAL_LLM_BASE_URL` | — | Any OpenAI-compatible server (vLLM, llama.cpp, LM Studio, a NIM LLM) |

Settings that are not secrets (endpoints, backend preference, LLM model and
temperature) can also be set in **Administration**; they are stored in SQLite and
overridden by the environment.

## Networking and proxies

* The app binds `0.0.0.0:8501` by default (`ADDRESS`/`PORT` override it) so it is
  reachable across interfaces, which is required for remote previews.
* `.streamlit/config.toml` sets `enableCORS=false` and `enableXsrfProtection=false`.
  This is what allows embedding behind a reverse proxy or a preview URL that
  terminates TLS elsewhere. **If you expose the console to an untrusted network,
  turn XSRF protection back on** and terminate TLS at your proxy.
* For nginx/Apache, proxy both HTTP and the WebSocket upgrade — Streamlit is a
  websocket app and a plain HTTP proxy will appear to hang.
* `server.enableWebsocketCompression` is off in the shipped config; leave it off if
  your proxy is unhappy with compressed frames.

## State, backups and cleanup

| Path | Contents |
| --- | --- |
| `.runtime/console.db` | conversations, runs, provenance, call log, settings |
| `.runtime/artifacts/<run_id>/` | exported PDB/SDF/CSV/JSON/Markdown files |

Both are git-ignored. Back them up together — a run's provenance refers to its
artifacts by directory name. To reset the console completely, stop it and delete
`.runtime/`; nothing else is written outside the repository.

`Administration → Performance → Storage` shows row counts and database size, and
**History → Database** has the same numbers with the resolved path.

## Security notes

* The console has **no authentication**. It is a single-user research tool: put it
  behind your own auth (VPN, SSO proxy, firewall) before sharing it.
* API keys entered in Administration are stored in SQLite in plain text. Prefer
  environment variables; the field is deliberately a password input and the value
  is never logged or exported.
* Jobs run in-process with a two-worker thread pool. Several users pointing at one
  instance will queue behind each other; give each user their own instance (or
  their own `BIONEMO_CONSOLE_HOME`) if that matters.
* Uploads are capped at 200 MB (`maxUploadSize`), which is far above any structure
  or library this console expects.

## Upgrading

```bash
git pull
bash scripts/setup.sh          # idempotent: updates deps and checks the viewer
# restart the console; the SQLite schema migrates itself on first open
```

The catalogue of NIMs (`bionemo_web/nims/catalog.py`) is the file to review when
NVIDIA changes an endpoint or ships a new model: add a `NimSpec`, and the model
appears in the chat router, the workbench dropdowns, the Administration endpoint
list and the launcher commands with no other change.
