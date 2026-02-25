"""
EDA and cleaning pipeline for dysarthria speech datasets (UASpeech, RAWDysPeech, etc.).
Pairs audio with transcriptions, filters by duration and CPS outliers, exports metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .uaspeech_processor import _parse_mlf, _read_audio_meta

logger = logging.getLogger(__name__)

AUDIO_SUBDIR = "audio/original"
MLF_SUBDIR = "mlf"
MLF_FILENAME_PATTERN = "{speaker}_word.mlf"


def _discover_uaspeech(audio_dir: Path, mlf_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speaker_dir in sorted(audio_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        mlf_path = mlf_dir / speaker_id / MLF_FILENAME_PATTERN.format(speaker=speaker_id)
        mlf_map = _parse_mlf(mlf_path) if mlf_path.exists() else {}
        if mlf_path.exists():
            logger.info("Speaker %s: %d transcriptions in MLF", speaker_id, len(mlf_map))
        else:
            logger.warning("MLF not found for speaker %s: %s", speaker_id, mlf_path)
        for wav_path in speaker_dir.glob("*.wav"):
            if wav_path.name.startswith("._"):
                continue
            basename = wav_path.stem
            transcript = mlf_map.get(basename, "")
            rows.append({
                "path": str(wav_path.resolve()),
                "speaker_id": speaker_id,
                "basename": basename,
                "transcript": transcript,
            })
    return rows


def _discover_flat(dataset_root: Path, mlf_root: Path) -> list[dict[str, Any]]:
    mlf_dir = mlf_root / MLF_SUBDIR
    cache: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for wav_path in dataset_root.rglob("*.wav"):
        if wav_path.name.startswith("._") or not wav_path.suffix.lower() == ".wav":
            continue
        basename = wav_path.stem
        parts = basename.split("_")
        speaker_id = parts[0] if parts else "unknown"
        if speaker_id not in cache:
            mlf_path = mlf_dir / speaker_id / MLF_FILENAME_PATTERN.format(speaker=speaker_id)
            cache[speaker_id] = _parse_mlf(mlf_path) if mlf_path.exists() else {}
        transcript = cache[speaker_id].get(basename, "")
        rows.append({
            "path": str(wav_path.resolve()),
            "speaker_id": speaker_id,
            "basename": basename,
            "transcript": transcript,
        })
    if cache and not any(cache.values()):
        logger.warning("No MLF files found under %s; all transcripts empty", mlf_dir)
    else:
        logger.info("Flat discovery: %d files, %d speakers", len(rows), len(cache))
    return rows


class DysarthriaProcessor:
    """
    Single pipeline for dysarthria datasets: discovery, duration/CPS filtering, plots, metadata export.
    Supports UASpeech layout (audio/original/<Speaker>/) and flat layout (any subdirs with .wav).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        mlf_root: str | Path | None = None,
        layout: Literal["uaspeech", "flat"] = "uaspeech",
        min_duration_sec: float = 3.0,
        cps_method: str = "iqr",
        cps_z_threshold: float = 3.0,
        cps_iqr_mult: float = 1.5,
        output_dir: str | Path | None = None,
        n_jobs: int | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.mlf_root = Path(mlf_root) if mlf_root else self.dataset_root
        self.layout = layout.lower()
        self.min_duration_sec = min_duration_sec
        self.cps_method = cps_method.lower()
        self.cps_z_threshold = cps_z_threshold
        self.cps_iqr_mult = cps_iqr_mult
        self.output_dir = Path(output_dir) if output_dir else self.dataset_root.parent / "pipeline_output"
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)

        if self.layout == "uaspeech":
            self.audio_dir = self.dataset_root / AUDIO_SUBDIR
            self.mlf_dir = self.mlf_root / MLF_SUBDIR
        else:
            self.audio_dir = self.dataset_root
            self.mlf_dir = self.mlf_root / MLF_SUBDIR

        self._manifest: pd.DataFrame | None = None
        self._cleaned: pd.DataFrame | None = None

    def discover(self) -> pd.DataFrame:
        """Pair each .wav with its transcription. Returns DataFrame with path, speaker_id, basename, transcript."""
        if self.layout == "uaspeech":
            if not self.audio_dir.is_dir():
                logger.error("Audio directory not found: %s", self.audio_dir)
                return pd.DataFrame()
            rows = _discover_uaspeech(self.audio_dir, self.mlf_dir)
        else:
            if not self.dataset_root.is_dir():
                logger.error("Dataset root not found: %s", self.dataset_root)
                return pd.DataFrame()
            rows = _discover_flat(self.dataset_root, self.mlf_root)
        df = pd.DataFrame(rows)
        logger.info("Discovered %d audio files", len(df))
        return df

    def _compute_audio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        paths = df["path"].tolist()
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            futures = {ex.submit(_read_audio_meta, p): p for p in paths}
            for fut in as_completed(futures):
                path, duration, sr = fut.result()
                results.append({"path": path, "duration": duration, "sample_rate": sr})
        meta = pd.DataFrame(results)
        df = df.merge(meta, on="path", how="left")
        df["char_count"] = df["transcript"].str.len()
        df["cps"] = np.where(df["duration"] > 0, df["char_count"] / df["duration"], np.nan)
        return df

    def build_manifest(self) -> pd.DataFrame:
        """Discover, compute metrics, filter by min_duration_sec. Sets self._manifest."""
        df = self.discover()
        if df.empty:
            return df
        df = self._compute_audio_metrics(df)
        logger.info("Dropping rows with duration < %s sec", self.min_duration_sec)
        valid = df[df["duration"] >= self.min_duration_sec].copy()
        logger.info("Duration filter: kept %d, dropped %d", len(valid), len(df) - len(valid))
        self._manifest = valid
        return self._manifest

    def _cps_outlier_mask(self, series: pd.Series) -> pd.Series:
        clean = series.dropna()
        if len(clean) < 4:
            return pd.Series(True, index=series.index)
        if self.cps_method == "zscore":
            z = np.abs(stats.zscore(clean))
            keep_index = clean.index[z <= self.cps_z_threshold]
            return series.index.isin(keep_index)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        low = q1 - self.cps_iqr_mult * iqr
        high = q3 + self.cps_iqr_mult * iqr
        return (series >= low) & (series <= high)

    def filter_cps_outliers(self) -> pd.DataFrame:
        """Remove CPS outliers from self._manifest. Sets self._cleaned."""
        if self._manifest is None or self._manifest.empty:
            logger.warning("No manifest; run build_manifest() first.")
            return pd.DataFrame()
        keep = self._cps_outlier_mask(self._manifest["cps"])
        self._cleaned = self._manifest.loc[keep].copy()
        logger.info("CPS outlier filter (%s): kept %d, removed %d", self.cps_method, len(self._cleaned), (~keep).sum())
        return self._cleaned

    def plot_distributions(self, prefix: str = "dataset") -> list[Path]:
        """Plot duration and CPS distributions under output_dir/reports. Returns saved figure paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            return saved
        for col, title, xlabel in [
            ("duration", "Duration distribution", "Duration (s)"),
            ("cps", "Character density (CPS) distribution", "Characters per second"),
        ]:
            plot_df = df[[col]].dropna()
            if len(plot_df) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=plot_df, x=col, kde=True, ax=ax, bins=min(80, max(20, len(plot_df) // 20)))
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            fpath = reports_dir / f"{prefix}_{col}_distribution.png"
            fig.tight_layout()
            fig.savefig(fpath, dpi=150)
            plt.close(fig)
            saved.append(fpath)
            logger.info("Saved %s", fpath)
        return saved

    def sample_for_review(self, n: int, random_state: int | None = 42) -> pd.DataFrame:
        """Random sample of N records for manual inspection; logs paths and metrics."""
        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            return pd.DataFrame()
        n = min(n, len(df))
        sample = df.sample(n=n, random_state=random_state)
        logger.info("Sample of %d records for manual review:", n)
        for _, row in sample.iterrows():
            logger.info(
                "  path=%s | speaker=%s | duration=%.2fs | sr=%s | chars=%s | cps=%.2f",
                row["path"], row["speaker_id"], row["duration"], row["sample_rate"], row["char_count"], row["cps"],
            )
        return sample

    def save_cleaned_manifest(self, filename: str = "metadata.csv") -> Path:
        """Save cleaned manifest to output_dir/data/<filename>."""
        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            raise RuntimeError("No manifest to save; run build_manifest() and optionally filter_cps_outliers() first.")
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / filename
        df.to_csv(out_path, index=False)
        logger.info("Saved manifest to %s (%d rows)", out_path, len(df))
        return out_path

    def run(
        self,
        plot_prefix: str = "dataset",
        manifest_filename: str = "metadata.csv",
    ) -> pd.DataFrame:
        """Full pipeline: discover -> duration filter -> CPS filter -> plots -> save manifest. Returns cleaned DataFrame."""
        self.build_manifest()
        self.filter_cps_outliers()
        self.plot_distributions(prefix=plot_prefix)
        self.save_cleaned_manifest(filename=manifest_filename)
        return self._cleaned
