# NirmanNetra AI

NirmanNetra AI is a geospatial computer-vision and municipal inspection
decision-support platform. It combines imagery analysis, municipal records and
human-controlled case review to identify potential structural changes.

The system does **not** declare construction illegal, issue notices or trigger
penalties. All operational recommendations require human review.

## Capabilities

- Deterministic synthetic municipal and geospatial datasets
- Raster metadata validation and image-quality assessment
- Before/after geospatial alignment and visual registration
- Building-footprint segmentation and polygon extraction
- Bitemporal added, removed and modified footprint detection
- Parcel, approved-plan and time-aware permit comparison
- Complaint deduplication and explainable risk scoring
- Inspector case lifecycle, evidence integrity and audit trails
- FastAPI application, Streamlit dashboard and PostgreSQL/PostGIS persistence
- Versioned model artifacts, municipal rules and Alembic migrations

## Processing flow

```text
Imagery ingestion → quality checks → registration → segmentation
→ change detection → parcel/permit evaluation → risk scoring
→ human-controlled inspection case
```

Large raster and derived artifacts remain in object storage. PostgreSQL stores
their metadata, checksums, relationships, geometries and audit records.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop for the container workflow

## Quick start

Clone and install all dependencies:

```powershell
git clone https://github.com/priyanshu141ai/Nirman-Netra-AI.git
cd Nirman-Netra-AI
uv sync --all-groups
```

Start the API:

```powershell
uv run uvicorn nirman_netra.api.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```powershell
$env:DASHBOARD_API_URL="http://127.0.0.1:8000"
uv run streamlit run src/nirman_netra/dashboard/app.py --server.port 8501
```

Open:

- Dashboard: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- Readiness: http://127.0.0.1:8000/health/ready

## Docker Compose

```powershell
$env:POSTGRES_PASSWORD="replace-with-a-strong-password"
docker compose up --build -d
docker compose ps
```

Default local endpoints:

- API documentation: http://127.0.0.1:18000/docs
- Dashboard: http://127.0.0.1:18502
- PostgreSQL: `127.0.0.1:15432`

Stop services while preserving volumes:

```powershell
docker compose down
```

## Synthetic demo

Run the deterministic end-to-end demonstration:

```powershell
uv run python -c "from pathlib import Path; from nirman_netra.application.demo import run_synthetic_demo; print(run_synthetic_demo(Path('data/demo')).model_dump_json(indent=2))"
```

Generate a machine-readable synthetic municipal dataset:

```powershell
uv run generate-synthetic --scenario approved_extension --seed 42 --crs EPSG:32643 --output data/synthetic
```

Supported scenarios include approved extensions, unapproved footprint
expansion, partial demolition, parcel encroachment, duplicate complaints,
low-quality imagery metadata and active or expired permits.

## Runtime configuration

Copy `.env.example` to `.env` and adjust values for the environment. Important
settings include:

- `DATABASE_URL`
- `USE_DATABASE_PERSISTENCE`
- `DATABASE_GEOMETRY_SRID`
- `OBJECT_STORAGE_ORIGINAL_PATH`
- `OBJECT_STORAGE_DERIVED_PATH`
- `MODEL_ARTIFACT_PATH`
- `MODEL_REQUIREMENT_PATH`
- `MUNICIPAL_CONTEXT_PATH`

Production processing requires explicit, compatible model and municipal-context
resources. Missing or incompatible resources never trigger a silent fallback.
Container paths must be mounted or otherwise readable inside the API container.

## API overview

The versioned `/api/v1` API supports:

- imagery assets and before/after pairs
- processing jobs and change results
- parcel and inspection-case lookup
- inspector assignment, review and reinspection
- approved-model and municipal-rule catalogues
- data-quality status

Use `/docs` for the complete request and response schemas.

## Development checks

```powershell
uv run pytest
uv run ruff check src tests migrations
uv run mypy src tests
uv run alembic upgrade head --sql
docker compose config --quiet
```

## Project structure

```text
src/nirman_netra/
├── api/                 FastAPI routes and contracts
├── application/         End-to-end orchestration and jobs
├── cases/               Case, evidence and audit workflow
├── change_detection/    Bitemporal change analysis
├── data/                Synthetic municipal data generation
├── dashboard/           Streamlit dashboard and API client
├── imagery/             Raster quality and registration
├── persistence/         PostgreSQL/PostGIS adapters
├── risk/                Parcel, permit, complaint and risk logic
└── segmentation/        Segmentation dataset, inference and artifacts
```

Detailed execution commands are also available in
[`RUN_INSTRUCTIONS.md`](RUN_INSTRUCTIONS.md).
