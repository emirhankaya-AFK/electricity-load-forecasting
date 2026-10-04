"""Command Line Interface (CLI) for electricity load forecasting."""

import argparse
import json
import logging
import sys

from electricity_load_forecasting.config import load_config
from electricity_load_forecasting.pipeline import PipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli")


def cmd_run(args: argparse.Namespace) -> int:
    """Run full pipeline."""
    cfg = load_config(args.config)
    if args.subset_hours is not None:
        cfg.data.subset_hours = args.subset_hours

    logger.info("Executing forecasting pipeline (offline=%s)...", args.offline)
    runner = PipelineRunner(config=cfg)
    results = runner.run(offline_mode=args.offline, save_reports=True)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Hourly Samples : {results['dataset_info']['total_hourly_samples']}")
    print(f"Train Samples        : {results['dataset_info']['train_samples']}")
    print(f"Validation Samples   : {results['dataset_info']['val_samples']}")
    print(f"Test Samples         : {results['dataset_info']['test_samples']}")
    print("-" * 60)
    print("Model Performance (Test Set):")
    print(
        f"  Seasonal-Naive Baseline MAE : {results['models']['seasonal_naive_baseline']['mae']:.4f} kW"
    )
    print(
        f"  HistGradientBoosting MAE    : {results['models']['hist_gradient_boosting']['mae']:.4f} kW"
    )
    print(
        f"  HistGradientBoosting RMSE   : {results['models']['hist_gradient_boosting']['rmse']:.4f} kW"
    )
    print(
        f"  HistGradientBoosting R2     : {results['models']['hist_gradient_boosting']['r2']:.4f}"
    )
    print(
        f"  Improvement Over Baseline   : {results['models']['improvement_vs_baseline_mae_pct']:.2f}%"
    )
    print("-" * 60)
    print("Conformal Prediction Intervals (Test Set):")
    print(f"  Nominal Coverage   : {results['conformal_intervals']['nominal_coverage'] * 100:.1f}%")
    print(
        f"  Empirical Coverage : {results['conformal_intervals']['empirical_coverage'] * 100:.2f}%"
    )
    print(f"  Mean Width         : {results['conformal_intervals']['mean_width']:.4f} kW")
    print(f"  Winkler Score      : {results['conformal_intervals']['winkler_score']:.4f}")
    print("-" * 60)
    print("Residual Anomaly Detection (Test Set):")
    print(f"  Anomaly Threshold   : {results['anomaly_detection']['threshold']:.4f} kW")
    print(
        f"  Natural Test Flags  : {results['anomaly_detection']['natural_test_anomalies_detected']}"
    )
    synth = results["anomaly_detection"]["synthetic_benchmark"]
    print(
        f"  Synthetic Benchmark : Precision={synth['precision']:.4f}, Recall={synth['recall']:.4f}, F1={synth['f1']:.4f} [SYNTHETIC]"
    )
    print("-" * 60)
    drift = results["drift_monitoring"]
    print(
        f"Drift Monitoring (PSI) : Overall Status = {drift['overall_status']}, Mean PSI = {drift['mean_psi']:.4f}"
    )
    print("=" * 60 + "\n")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve FastAPI application with Uvicorn."""
    import uvicorn

    cfg = load_config(args.config)
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port

    logger.info("Starting FastAPI service on %s:%d ...", host, port)
    uvicorn.run("electricity_load_forecasting.api:app", host=host, port=port, reload=False)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute latency benchmark."""
    from benchmark import run_benchmarks

    logger.info("Running system latency benchmarks...")
    results = run_benchmarks(n_iterations=args.iterations)
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60 + "\n")
    return 0


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="electricity-forecast",
        description="Electricity Load Forecasting CLI",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    parser_run = subparsers.add_parser(
        "run", help="Run full training, calibration, and evaluation pipeline"
    )
    parser_run.add_argument(
        "--offline", action="store_true", help="Run offline using deterministic synthetic data"
    )
    parser_run.add_argument("--subset-hours", type=int, default=None, help="Override subset hours")
    parser_run.set_defaults(func=cmd_run)

    # Subcommand: serve
    parser_serve = subparsers.add_parser("serve", help="Start FastAPI REST server")
    parser_serve.add_argument("--host", type=str, default=None, help="Host address")
    parser_serve.add_argument("--port", type=int, default=None, help="Port number")
    parser_serve.set_defaults(func=cmd_serve)

    # Subcommand: benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run latency benchmarks")
    parser_bench.add_argument(
        "--iterations", type=int, default=500, help="Number of benchmark iterations"
    )
    parser_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
