# Run NirmanNetra AI

## Prerequisites

- Python 3.11
- `uv`
- Docker Desktop only for the container workflow

## Local workflow (PowerShell)

Install dependencies:

```powershell
uv sync --all-groups
```

Start the API in terminal 1:

```powershell
uv run uvicorn nirman_netra.api.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in terminal 2:

```powershell
$env:DASHBOARD_API_URL="http://127.0.0.1:8000"
uv run streamlit run src/nirman_netra/dashboard/app.py --server.address=127.0.0.1 --server.port=8501
```

Open:

- Dashboard: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- API readiness: http://127.0.0.1:8000/health/ready

Stop each process with `Ctrl+C`.

## Deterministic synthetic demo

```powershell
uv run python -c "from pathlib import Path; from nirman_netra.application.demo import run_synthetic_demo; print(run_synthetic_demo(Path('data/demo')).model_dump_json(indent=2))"
```

The output contains approved-change, permit-mismatch, duplicate-complaint,
low-registration-quality and evidence-integrity scenarios. Results always require
human review and never represent an automatic legal verdict.

## Docker Compose workflow

Start Docker Desktop, then run:

```powershell
$env:POSTGRES_PASSWORD="replace-with-a-strong-password"
docker compose up --build
```

Compose starts PostgreSQL/PostGIS, runs Alembic migrations, then starts the API
and dashboard. Check status with:

```powershell
docker compose ps
docker compose logs -f api dashboard
```

Stop containers while preserving database and object-storage volumes:

```powershell
docker compose down
```

Remove volumes only when all local container data may be discarded:

```powershell
docker compose down --volumes
```
