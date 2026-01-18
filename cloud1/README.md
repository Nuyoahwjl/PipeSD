# Cloud (verification service)

FastAPI service that hosts the target model, verifies speculative tokens, and reports GPU power-integral metrics.

## Key files
- `src/speculative_server.py`: FastAPI app + model execution + verify flow
- `src/util.py`: shared utilities
- `install.sh`: dependency
- `hfd.sh`: model downloader

## Directory layout
- `pre_models/`: target-model GGUF files (downloaded or placed manually)

## Running the service
From the repo root `cloud/` directory:
```bash
python -m src.speculative_server
```

The service binds to port `8000` by default (see `APP_PORT` in `src/speculative_server.py`).

## API endpoints (high level)
- `POST /init`: initialize a task with a token prefix
- `POST /propose`: send speculative tokens and/or trigger verification
- `POST /exit`: finalize a task and return aggregated metrics
- `GET /health`: service status
