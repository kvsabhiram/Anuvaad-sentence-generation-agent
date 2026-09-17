"""CLI entrypoint for the English -> Telugu translation pipeline."""

from __future__ import annotations

import argparse

from pipeline.orchestrator import run


def main() -> None:
    parser = argparse.ArgumentParser(description="English to Telugu translation pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the pipeline config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N sentences. Use this for a pilot run before "
            "committing to the full corpus -- it exercises every stage end to end "
            "(translate, rule checks, judge, embedding, QC, export) at small cost."
        ),
    )
    args = parser.parse_args()
    run(config_path=args.config, limit=args.limit)


if __name__ == "__main__":
    main()
