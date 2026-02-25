#!/usr/bin/env python3
"""
Run EDA and cleaning for dysarthria datasets (UASpeech, RAWDysPeech, etc.).
Use --dataset <name> from datasets.yaml or --dataset-root with optional --mlf-root.
Outputs: reports/ (plots), data/<manifest> (default metadata.csv).
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import DysarthriaProcessor
from src.dataset_config import get_dataset_config, list_dataset_names

CONFIG_PATH = PROJECT_ROOT / "datasets.yaml"


def _detect_layout(dataset_root: Path) -> str:
    audio_original = dataset_root / "audio" / "original"
    if audio_original.is_dir():
        return "uaspeech"
    return "flat"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA and cleaning pipeline for dysarthria speech datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="NAME",
        help="Named dataset from datasets.yaml (e.g. uaspeech, rawdyspeech). Overrides --dataset-root if set.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to dataset root (used if --dataset not set). Layout auto-detected unless --layout set.",
    )
    parser.add_argument(
        "--mlf-root",
        type=Path,
        default=None,
        help="Path to MLF root (dir containing mlf/<Speaker>/). For flat layout required unless using --dataset.",
    )
    parser.add_argument(
        "--layout",
        choices=["uaspeech", "flat"],
        default=None,
        help="Dataset layout: uaspeech (audio/original/<Speaker>/) or flat (any subdirs with .wav). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory for reports/ and data/",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="metadata.csv",
        metavar="FILE",
        help="Output manifest filename in data/",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum duration in seconds",
    )
    parser.add_argument(
        "--cps-method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="CPS outlier method",
    )
    parser.add_argument(
        "--cps-z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold when cps-method=zscore",
    )
    parser.add_argument(
        "--cps-iqr-mult",
        type=float,
        default=1.5,
        help="IQR multiplier when cps-method=iqr",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Number of records to sample for manual review (0 = skip)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for audio reading",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List dataset names from datasets.yaml and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if args.list_datasets:
        names = list_dataset_names(CONFIG_PATH)
        if not names:
            print("No datasets in datasets.yaml (or file missing).")
        else:
            print("Datasets:", ", ".join(names))
        sys.exit(0)

    dataset_root: Path | None = None
    mlf_root: Path | None = None
    layout: str | None = args.layout
    manifest_name = args.manifest
    plot_prefix = "dataset"

    if args.dataset:
        cfg = get_dataset_config(CONFIG_PATH, args.dataset, PROJECT_ROOT)
        if not cfg:
            logger.error("Unknown dataset '%s'. Add it to datasets.yaml or use --list-datasets.", args.dataset)
            sys.exit(1)
        dataset_root = cfg["root"]
        mlf_root = cfg["mlf_root"]
        layout = cfg["layout"]
        manifest_name = cfg.get("output_manifest", args.manifest)
        plot_prefix = args.dataset
        logger.info("Using dataset '%s': root=%s, mlf_root=%s, layout=%s", args.dataset, dataset_root, mlf_root, layout)
    elif args.dataset_root:
        dataset_root = args.dataset_root.resolve()
        mlf_root = args.mlf_root.resolve() if args.mlf_root else dataset_root
        if not layout:
            layout = _detect_layout(dataset_root)
            logger.info("Auto-detected layout: %s", layout)
        if layout == "flat" and (not args.mlf_root and not args.dataset):
            logger.warning("Flat layout usually needs --mlf-root (path to UASpeech) for transcripts.")
    else:
        cfg = get_dataset_config(CONFIG_PATH, "uaspeech", PROJECT_ROOT)
        if cfg:
            dataset_root = cfg["root"]
            mlf_root = cfg["mlf_root"]
            layout = cfg["layout"]
            manifest_name = cfg.get("output_manifest", args.manifest)
            plot_prefix = "uaspeech"
            logger.info("Defaulting to dataset 'uaspeech' from config: %s", dataset_root)
        if not dataset_root or not dataset_root.is_dir():
            logger.error("Set --dataset <name> or --dataset-root PATH. Use --list-datasets to see names.")
            sys.exit(1)

    if not dataset_root or not dataset_root.is_dir():
        logger.error("Dataset root is not a directory: %s", dataset_root)
        sys.exit(1)

    processor = DysarthriaProcessor(
        dataset_root=dataset_root,
        mlf_root=mlf_root,
        layout=layout,
        min_duration_sec=args.min_duration,
        cps_method=args.cps_method,
        cps_z_threshold=args.cps_z_threshold,
        cps_iqr_mult=args.cps_iqr_mult,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )

    processor.run(plot_prefix=plot_prefix, manifest_filename=manifest_name)

    if args.sample > 0:
        processor.sample_for_review(n=args.sample)

    logger.info("Done. Reports: %s/reports/  Manifest: %s/data/%s", args.output_dir, args.output_dir, manifest_name)


if __name__ == "__main__":
    main()
