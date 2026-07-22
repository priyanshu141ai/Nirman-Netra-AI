"""Command-line entry point for idempotent synthetic generation."""

import argparse
from pathlib import Path

from nirman_netra.config import load_settings
from nirman_netra.data.contracts import GenerationConfig, Scenario
from nirman_netra.data.generator import generate_dataset
from nirman_netra.geospatial import validate_crs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic municipal data")
    parser.add_argument(
        "--scenario", required=True, choices=[scenario.value for scenario in Scenario]
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--crs")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    configured_crs = args.crs or load_settings().default_processing_crs
    if configured_crs is None:
        parser.error("--crs or DEFAULT_PROCESSING_CRS is required")
    config = GenerationConfig(
        scenario=Scenario(args.scenario),
        random_seed=args.seed,
        processing_crs=validate_crs(configured_crs),
    )
    manifest = generate_dataset(config, args.output)
    print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
