from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ForecastRequest, ForecastingAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-first automated forecasting agent")
    parser.add_argument("--csv", required=True, help="Input dataset CSV path")
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon in steps")
    parser.add_argument("--output", required=True, help="Output forecast CSV path")
    parser.add_argument("--time-column", default=None, help="Explicit time column name")
    parser.add_argument("--target-column", default=None, help="Explicit target column name")
    parser.add_argument("--series-column", default=None, help="Optional series/group column")
    parser.add_argument("--frequency", default=None, help="Optional frequency override")
    parser.add_argument("--interval-level", default=0.9, type=float, help="Prediction interval level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    request = ForecastRequest(
        csv_path=Path(args.csv),
        horizon=args.horizon,
        output_path=Path(args.output),
        time_column=args.time_column,
        target_column=args.target_column,
        series_column=args.series_column,
        frequency=args.frequency,
        interval_level=args.interval_level,
    )
    result = ForecastingAgent().run(request)
    print(result.summary)
    print()
    print("Model selection explanation:")
    print(result.selection_explanation)


if __name__ == "__main__":
    main()
