from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

AUDIO_SUBDIR = "audio/original"
MLF_SUBDIR = "mlf"
MLF_FILENAME_PATTERN = "{speaker}_word.mlf"
WAV_PATTERN = re.compile(r"^([A-Z0-9]+)_(B[123])_[A-Z0-9]+_(M[2-8])\.wav$", re.IGNORECASE)


def _parse_mlf(path: Path) -> dict[str, str]:
    """
    Parse a single MLF file into a mapping from utterance basename (no extension) to word.

    MLF format: each entry is "*/BASENAME.lab", next line is the word, then ".".
    """
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip() for line in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('"*/') and line.endswith('.lab"'):
            basename = line[3:-5]
            i += 1
            if i < len(lines):
                word = lines[i].strip()
                out[basename] = word
            i += 1
            if i < len(lines) and lines[i] == ".":
                i += 1
        else:
            i += 1
    return out


def _read_audio_meta(path: str) -> tuple[str, float, int]:
    """Read duration (seconds) and sample rate for one wav file. Returns (path, duration, sr)."""
    try:
        info = sf.info(path)
        duration = info.duration
        sr = info.samplerate
        return (path, duration, sr)
    except Exception as e:
        logger.warning("Could not read audio %s: %s", path, e)
        return (path, float("nan"), 0)


class UASpeechProcessor:
    """
    End-to-end processor for UASpeech: discovery, metrics, filtering, and manifest export.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        min_duration_sec: float = 3.0,
        cps_method: str = "iqr",
        cps_z_threshold: float = 3.0,
        cps_iqr_mult: float = 1.5,
        output_dir: str | Path | None = None,
        n_jobs: int | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.min_duration_sec = min_duration_sec
        self.cps_method = cps_method.lower()
        self.cps_z_threshold = cps_z_threshold
        self.cps_iqr_mult = cps_iqr_mult
        self.output_dir = Path(output_dir) if output_dir else self.dataset_root.parent / "pipeline_output"
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)

        self.audio_dir = self.dataset_root / AUDIO_SUBDIR
        self.mlf_dir = self.dataset_root / MLF_SUBDIR

        self._manifest: pd.DataFrame | None = None
        self._cleaned: pd.DataFrame | None = None

    def discover(self) -> pd.DataFrame:
        """
        Traverse dataset to pair each .wav with its transcription from MLF files.
        Returns a DataFrame with columns: path, speaker_id, basename, transcript.
        """
        rows: list[dict[str, Any]] = []
        if not self.audio_dir.is_dir():
            logger.error("Audio directory not found: %s", self.audio_dir)
            return pd.DataFrame()

        for speaker_dir in sorted(self.audio_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            mlf_path = self.mlf_dir / speaker_id / MLF_FILENAME_PATTERN.format(speaker=speaker_id)
            if not mlf_path.exists():
                logger.warning("MLF not found for speaker %s: %s", speaker_id, mlf_path)
                mlf_map = {}
            else:
                mlf_map = _parse_mlf(mlf_path)
                logger.info("Speaker %s: %d transcriptions in MLF", speaker_id, len(mlf_map))

            for wav_path in speaker_dir.glob("*.wav"):
                if wav_path.name.startswith("._"):
                    continue
                basename = wav_path.stem
                transcript = mlf_map.get(basename)
                if transcript is None:
                    logger.debug("No transcript for %s", wav_path)
                    transcript = ""
                rows.append({
                    "path": str(wav_path.resolve()),
                    "speaker_id": speaker_id,
                    "basename": basename,
                    "transcript": transcript,
                })

        df = pd.DataFrame(rows)
        logger.info("Discovered %d audio files with transcripts", len(df))
        return df

    def _compute_audio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add duration and sample_rate using parallel reads."""
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
        df["cps"] = np.where(
            df["duration"] > 0,
            df["char_count"] / df["duration"],
            np.nan,
        )
        return df

    def build_manifest(self) -> pd.DataFrame:
        """
        Discover pairs, compute duration/sample_rate/CPS, and filter by min_duration_sec.
        Sets self._manifest to the post-duration-filter DataFrame.
        """
        df = self.discover()
        if df.empty:
            return df
        df = self._compute_audio_metrics(df)
        logger.info("Metrics computed. Dropping rows with duration < %s sec", self.min_duration_sec)
        valid = df[df["duration"] >= self.min_duration_sec].copy()
        dropped = len(df) - len(valid)
        logger.info("Duration filter: kept %d, dropped %d", len(valid), dropped)
        self._manifest = valid
        return self._manifest

    def _cps_outlier_mask(self, series: pd.Series) -> pd.Series:
        """True = keep, False = outlier (garbage)."""
        clean = series.dropna()
        if len(clean) < 4:
            return pd.Series(True, index=series.index)
        if self.cps_method == "zscore":
            z = np.abs(stats.zscore(clean))
            threshold = self.cps_z_threshold
            keep_index = clean.index[z <= threshold]
            return series.index.isin(keep_index)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        low = q1 - self.cps_iqr_mult * iqr
        high = q3 + self.cps_iqr_mult * iqr
        return (series >= low) & (series <= high)

    def filter_cps_outliers(self) -> pd.DataFrame:
        """
        Remove CPS outliers (garbage) from self._manifest. Sets self._cleaned.
        """
        if self._manifest is None or self._manifest.empty:
            logger.warning("No manifest; run build_manifest() first.")
            return pd.DataFrame()
        keep = self._cps_outlier_mask(self._manifest["cps"])
        self._cleaned = self._manifest.loc[keep].copy()
        n_removed = (~keep).sum()
        logger.info("CPS outlier filter (%s): kept %d, removed %d", self.cps_method, len(self._cleaned), n_removed)
        return self._cleaned

    def plot_distributions(self, prefix: str = "uaspeech") -> list[Path]:
        """
        Plot duration and CPS distributions. Save figures under output_dir/reports.
        Returns paths to saved figures.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            logger.warning("No data to plot; run build_manifest (and optionally filter_cps_outliers) first.")
            return saved

        for col, title, xlabel in [
            ("duration", "Duration distribution", "Duration (s)"),
            ("cps", "Character density (CPS) distribution", "Characters per second"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_df = df[[col]].dropna()
            if len(plot_df) == 0:
                continue
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
        """
        Randomly sample N records from the cleaned set for manual inspection.
        Logs paths and metrics; returns the sample DataFrame.
        """
        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            logger.warning("No data to sample.")
            return pd.DataFrame()
        n = min(n, len(df))
        sample = df.sample(n=n, random_state=random_state)
        logger.info("Sample of %d records for manual review:", n)
        for _, row in sample.iterrows():
            logger.info(
                "  path=%s | speaker=%s | duration=%.2fs | sr=%s | chars=%s | cps=%.2f",
                row["path"],
                row["speaker_id"],
                row["duration"],
                row["sample_rate"],
                row["char_count"],
                row["cps"],
            )
        return sample

    def save_cleaned_manifest(self, filename: str = "cleaned_manifest.csv") -> Path:
        """Save the cleaned manifest (paths + metrics) to output_dir/data/filename."""
        df = self._cleaned if self._cleaned is not None else self._manifest
        if df is None or df.empty:
            raise RuntimeError("No manifest to save; run build_manifest (and optionally filter_cps_outliers) first.")
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / filename
        df.to_csv(out_path, index=False)
        logger.info("Saved cleaned manifest to %s (%d rows)", out_path, len(df))
        return out_path

    def run(
        self,
        plot_prefix: str = "uaspeech",
        manifest_filename: str = "cleaned_manifest.csv",
    ) -> pd.DataFrame:
        """
        Full pipeline: discover -> duration filter -> CPS outlier filter -> plots -> save manifest.
        Returns the final cleaned DataFrame.
        """
        self.build_manifest()
        self.filter_cps_outliers()
        self.plot_distributions(prefix=plot_prefix)
        self.save_cleaned_manifest(filename=manifest_filename)
        return self._cleaned
