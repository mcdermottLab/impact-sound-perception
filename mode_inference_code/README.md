# Fit modal parameters to impulse responses

`fit_example_impulse_responses.py` fits the ten-mode `ImpactSound` model to
observed impulse responses. It retains the waveform preprocessing, objective,
Adam settings, gradient clipping, and early-stopping rules.

## Installation and first run

Use Python 3.12 and install the dependencies from the repository root:

```bash
python -m pip install -r mode_inference_code/requirements.txt
python mode_inference_code/fit_example_impulse_responses.py
```

No positional arguments are required. Defaults search the included
`data/example_IRs/` recursively for `h.wav`, sort the paths, and fit the first
two files. The optimization loop limit is 100,000; the historical loop runs
from 1 to one less than this limit and may stop earlier.

For a quick input/output check (one optimization update per input):

```bash
python mode_inference_code/fit_example_impulse_responses.py --steps 2
```

To select a WAV explicitly:

```bash
python mode_inference_code/fit_example_impulse_responses.py bottle check \
  --input data/example_IRs/plastic_hard_soda_bottle_8_8_28/ch00/h.wav \
  --output-dir outputs/modal_fits --steps 1001
```

`bottle` and `check` label the run; they are not filenames. The output is
`outputs/modal_fits/bottle_check/`. Absolute input and output paths are printed
before fitting. All required directories are created automatically.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `run_name` | `example_IRs` | First output label: letters, numbers, underscores, hyphens, and periods |
| `experiment_name` | `demo` | Second output label |
| `--input` | Repository `data/example_IRs/` | WAV or directory searched recursively for `h.wav` |
| `--output-dir` | Repository `outputs/modal_fits/` | Parent of the run directory |
| `--num-sounds` | `2` | Maximum recordings to fit; a single input file fits once |
| `--start-index` | `0` | First index in the sorted input list |
| `--steps` | `100000` | Exclusive optimization-loop upper bound; at least 2 |
| `--help` | — | Show usage and defaults |

Default paths are based on the script location. Explicit relative paths use
the current working directory. Inputs must be **mono WAVs at 44,100 Hz**.
Original preprocessing starts at the selected peak, removes the mean,
normalizes, and truncates or zero-pads to 0.5 seconds. Resampling and stereo
downmixing are not applied.

## Saved results

`dirnames.csv` and `filenames.csv` list discovered input paths. Numbered
subdirectories correspond to zero-based input indices. Each may contain:

| Output | Contents and timing |
|---|---|
| `observed_1.wav` | Normalized preprocessed target, saved after the first update |
| `RT60.csv`, `Amp.csv`, `freq.csv`, `alpha.csv`, `gamma.csv` | Latest saved parameters, updated at steps 1, 501, 1001, … and selected early-stop conditions |
| `RT60/`, `Amp/`, `freq/`, `Alpha/`, `Gamma/` | Numbered parameter snapshots every 1,000 steps, e.g. `RT60/RT60_bottle1000.csv` |
| `loss.png` | Loss trace, refreshed every 100 steps |
| `output_<step>.wav` | Normalized synthesis at steps 1, 501, 1001, … unless the loss-based stopping condition is met first |

Files appear only if the corresponding save interval is reached. `RT60` is
modal decay time in seconds, `freq` is modal frequency in Hz, and `Amp` stores
amplitude parameters in dB. `alpha` and `gamma` are envelope parameters; see
`impact_model.py` and `sound_renderer.py` for their parameterization. The
top-level CSVs are the latest *saved* state, not necessarily the last step.

Reusing labels and an output directory overwrites matching files. Choose a
new label or directory for a separate run. Random initialization means repeated
runs need not produce identical fits. Historical variants are retained in
`extras/`; the entry point above is the supported portable example.
