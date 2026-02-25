# Dysarthria Speech EDA & Cleaning

Pipeline for dysarthria speech datasets: discover audio and transcriptions, filter by duration and CPS outliers, export `metadata.csv` and distribution plots. Supports UASpeech-style layout and flat folder layouts (e.g. RAWDysPeech).

---

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --list-datasets
python run_pipeline.py --dataset uaspeech
```

Results: `reports/<dataset>_*.png`, `data/metadata.csv`.

---

## How to add your own dataset

### 1. Choose layout

Your dataset is one of:

- **UASpeech layout** — audio in `audio/original/<SpeakerID>/` and transcripts in `mlf/<SpeakerID>/<SpeakerID>_word.mlf` (HTK MLF). Example: [UASpeech](https://www.isle.illinois.edu/sst/data/uaspeech/).
- **Flat layout** — `.wav` files in any subfolders (e.g. `0/`, `1/`), filenames like `CF02_B1_C10_M2.wav` (speaker = first segment). Transcripts must come from MLF files elsewhere (e.g. UASpeech `mlf/`).

### 2. Edit `datasets.yaml`

Add a new block under `datasets:` with a **unique name** (e.g. `my_dataset`):

```yaml
datasets:
  my_dataset:
    root: /absolute/path/to/your/audio/root
    mlf_root: null
    layout: uaspeech
    output_manifest: metadata.csv
```

| Field | Meaning |
|-------|--------|
| **root** | Path to the dataset root (where audio lives). For UASpeech layout this root must contain `audio/original/` and `mlf/`. For flat layout it is the folder that contains your `.wav` subdirs. |
| **mlf_root** | Path to the directory that contains `mlf/<SpeakerID>/<SpeakerID>_word.mlf`. Use `null` if transcripts live inside `root` (UASpeech layout). For flat layout, set this to the UASpeech root (or any path that has the `mlf/` tree). |
| **layout** | `uaspeech` or `flat` (see above). |
| **output_manifest** | Name of the output CSV in `data/` (e.g. `metadata.csv`). |

**Examples:**

- **UASpeech** (audio and MLF in one place):
  ```yaml
  my_uaspeech:
    root: /data/UASpeech
    mlf_root: null
    layout: uaspeech
    output_manifest: metadata.csv
  ```

- **Flat** (e.g. RAWDysPeech; transcripts from UASpeech):
  ```yaml
  my_raw:
    root: /data/RAWDysPeech
    mlf_root: /data/UASpeech
    layout: flat
    output_manifest: metadata.csv
  ```

### 3. Run

```bash
python run_pipeline.py --dataset my_dataset
```

Output: `reports/my_dataset_duration_distribution.png`, `reports/my_dataset_cps_distribution.png`, `data/metadata.csv`.

---

## How to run (without adding to config)

You can run without editing `datasets.yaml` by passing paths:

```bash
# UASpeech-style (layout auto-detected if audio/original exists)
python run_pipeline.py --dataset-root /path/to/UASpeech

# Flat layout (must pass MLF root and layout)
python run_pipeline.py --dataset-root /path/to/RAWDysPeech --mlf-root /path/to/UASpeech --layout flat
```

Optional: `--output-dir`, `--manifest`, `--min-duration`, `--cps-method`, `--sample N`, `--n-jobs`, `-v`. See `python run_pipeline.py --help`.

---

## Outputs

| Path | Description |
|------|-------------|
| `reports/<prefix>_duration_distribution.png` | Histogram of recording duration (after filters). |
| `reports/<prefix>_cps_distribution.png` | Histogram of characters per second (CPS). |
| `data/<output_manifest>` | Cleaned table: `path`, `speaker_id`, `basename`, `transcript`, `duration`, `sample_rate`, `char_count`, `cps`. |

Pipeline steps: discover audio + transcripts → filter duration ≥ 3 s → compute CPS → remove CPS outliers (IQR or Z-score) → plot → write CSV. Optional `--sample N` logs N random rows for manual check.

---

## Programmatic use

```python
from pathlib import Path
from src.processor import DysarthriaProcessor

processor = DysarthriaProcessor(
    dataset_root=Path("/path/to/audio/root"),
    mlf_root=Path("/path/to/mlf/root"),
    layout="flat",
    min_duration_sec=3.0,
    output_dir=Path("."),
)
processor.run(plot_prefix="mydata", manifest_filename="metadata.csv")
```
