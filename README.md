# Impact sound perception: code and data

Code and supporting data for:

**Intuitive knowledge of object acoustics enables perceptual separation of physical variables from impact sounds**

**Vinayak Agarwal\*, James Traer\*, Jeremy Schwartz, and Josh H. McDermott**

*Proceedings of the National Academy of Sciences* (2026).
[Paper: DOI 10.1073/pnas.2605876123](https://doi.org/10.1073/pnas.2605876123)

\* Vinayak Agarwal and James Traer contributed equally.

This repository contains the probabilistic impact-sound model, modal fitting
code, human experiment data and analyses, main and supplementary figure
scripts, and saved observer results. Start with the supplied data and statistics
to reproduce the analyses; use the fitting and synthesis examples to explore
the generative model.

## Abstract

Upon hearing objects collide, humans can estimate physical attributes such as
material and mass. Although the physics of sound generation is well established,
the inverse problem that listeners solve—of inferring physical parameters from
sound—remains poorly understood. Classical accounts posit the use of acoustic
cues that correlate with physical variables but do not explain how humans might
distinguish multiple concurrent physical causes. To study this problem, we
measured impulse responses from hundreds of everyday objects, extracted their
resonances, and used statistics of the resonances to build a probabilistic
generative model of impact sounds. We then used the model to synthesize and
manipulate stimuli for experiments measuring judgments of physical variables.
Humans accurately judged object properties from collision sounds. However, when
both of the colliding objects varied, performance was impaired if the distribution
of object resonances deviated from those measured in real-world objects. The
results suggest that humans have internalized the statistical regularities of
real-world object acoustics, and use this knowledge to separate the acoustic
contributions of the distinct objects in the world that give rise to sounds.

## Contents

- [Requirements and installation](#requirements-and-installation)
- [Reproduce the paper statistics](#reproduce-the-paper-statistics)
- [Generate figures](#generate-figures)
- [Fit modal parameters to an impulse response](#fit-modal-parameters-to-an-impulse-response)
- [Synthesize impact sounds](#synthesize-impact-sounds)
- [Replay the damping observer](#replay-the-damping-observer)
- [Data and experiment samples](#data-and-experiment-samples)
- [Repository layout](#repository-layout)
- [Reproducibility and scope](#reproducibility-and-scope)
- [Citation, attribution, and contact](#citation-attribution-and-contact)

## Requirements and installation

Clone or download the repository, keeping the directory structure intact:

```bash
git clone https://github.com/mcdermottLab/impact-sound-perception.git
cd impact-sound-perception
```

### MATLAB

The analyses require MATLAB and the **Statistics and Machine Learning Toolbox**.
Audio synthesis and several figure scripts also require the **Signal Processing
Toolbox**. Verification used **MATLAB R2025b**; older releases may require syntax
adjustments. Run `setup_paths` from the repository root before entering a script
directory to add the bundled helpers and synthesis functions.

### Python

Use a separate environment. **Python 3.12** is the tested interpreter for the
modal-fitting example; its pinned PyTorch/TorchAudio pair supports Python
3.9–3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

For modal fitting, install the additional requirements:

```bash
python -m pip install -r mode_inference_code/requirements.txt
```

The fitting requirements pin matching PyTorch and TorchAudio versions to keep
the audio-writing API reproducible. The cochleagram utilities are included.
CPU execution is supported; a GPU is not required.

## Reproduce the paper statistics

Open MATLAB in the repository root:

```matlab
setup_paths
results = run_paper_statistics(10000, fullfile(pwd, 'outputs', 'statistics'));
export_exp3c_exp4_demographics(fullfile(pwd, 'outputs', 'statistics'));
```

This writes:

- `supplementary_statistics.csv`: the consolidated 40-comparison statistics
  table with effect estimates, 95% confidence intervals, p-values, sample sizes,
  and diagnostic bounds.
- Per-experiment MAT files and `all_stats.mat`: detailed results, including
  bootstrap and permutation distributions.
- `Exp3c_Exp4_demographics_for_Josh.csv`: the demographics summary for the
  original Experiment 3c and 4 samples.

Reference outputs are supplied in [`stats_results/`](stats_results/).
Omitting the output-directory argument writes into that folder instead.
Analyses use 10,000 resamples by default and fixed local random streams.
P-values and confidence intervals in the CSV are unadjusted for multiple
comparisons. See [`stats_results/README.md`](stats_results/README.md) for the
table schema and comparison definitions.

Experiments 3a/3b use 80%-correct thresholds; Experiment 4 uses d′ = 1 thresholds
from pooled hit and false-alarm rates. Statistical threshold inference uses
linear interpolation at the first upward criterion crossing, paired participant
bootstraps, and within-participant condition permutations. Out-of-range
thresholds retain censoring bounds. The Experiment 3a inverted-distribution
comparison has a bounded p-value, and the corresponding Experiment 3b confidence
interval has an unbounded upper limit. Read the `Status`, `PValueLowerBound`,
and `PValueUpperBound` columns when interpreting these rows.

## Generate figures

From the repository root in MATLAB:

```matlab
setup_paths
run_paper_figures('/absolute/path/to/.venv/bin/python')
```

Use your Python environment's actual executable path. The driver writes SVG
files under `Figures/all_figs/`. Figure 1 is a conceptual illustration. Figure 6
samples many modal configurations and takes substantially longer than the other
scripts. The optional spectral-flatness panel is separate from the default driver.

For a single figure, run `setup_paths`, then enter that figure's directory:

```matlab
cd(fullfile('Figures', 'Figure 10'))
plot_figure_10acd_damping_discrimination
```

Folder and script panel suffixes identify the original standalone panels;
the published composite figure may use different panel letters. The scripts
retain the paper's plotting and error-bar conventions. Those display
calculations are separate from the consolidated statistical inference above.

Supplementary scripts are in [`supplementary_figures/`](supplementary_figures/).
Run them from their own directories after `setup_paths`. For example, starting
from the repository root:

```matlab
cd(fullfile('supplementary_figures', 'supplementary figure 1', 'original_scripts'))
plot_supplementary_figure_01c_common_modes
plot_supplementary_figure_01d_orphan_modes
```

Outputs are `frequency_pairs_comparison.svg`, `frequency_pairs_histograms.svg`,
and `unpaired_modes_comparison.svg`. Required CSV and observed-audio inputs are
included.

## Fit modal parameters to an impulse response

From the repository root, with the fitting environment activated:

```bash
python mode_inference_code/fit_example_impulse_responses.py
```

This fits the first two included `h.wav` examples, in sorted path order, using
the original optimization settings and stopping rules. It prints the selected
inputs and absolute output directory. The default run directory is
`outputs/modal_fits/example_IRs_demo/`. Full optimization can be lengthy.

To select a single recording and a short run:

```bash
python mode_inference_code/fit_example_impulse_responses.py bottle check \
  --input data/example_IRs/plastic_hard_soda_bottle_8_8_28/ch00/h.wav \
  --output-dir outputs/modal_fits --steps 1001
```

The optional positional arguments (`bottle` and `check`) are **output labels**,
not file paths. `--input` accepts a WAV or a directory containing `h.wav` files.
Inputs must be mono, 44,100-Hz recordings. Default paths are based on the script
location; explicit relative paths are relative to your current directory.

Numbered output directories contain the preprocessed observed waveform,
parameter CSVs, and optimization diagnostics. `dirnames.csv` and `filenames.csv`
map input indices to source recordings. Checkpoints go into `RT60/`, `Amp/`,
`freq/`, `Alpha/`, and `Gamma/`; required directories are created automatically,
including when reusing a run directory.

See [`mode_inference_code/README.md`](mode_inference_code/README.md) for all
options, output definitions, and checkpoint timing.

## Synthesize impact sounds

From the repository root in MATLAB:

```matlab
setup_paths
cd experimental_stimuli
generate_exp3a_stimuli
```

Generators use the included modal statistics, stiffness table, and synthesis
helpers. Experiment 0 also uses `data/recorded_impacts/`. They generate large
audio sets beneath their configured output paths. Synthesis is stochastic;
newly generated WAVs need not match historical stimuli sample for sample.

Some historical generators depend on legacy synthesis variants that are not
included. Supplied participant-data analyses and saved-output plotting are the
primary reproduction paths; inspect a generator's dependencies before starting
a large regeneration run.

## Replay the damping observer

From the repository root in MATLAB:

```matlab
setup_paths
mod_ans_all = replay_damping_observer;
```

The replay checks all **105,000 saved trials** against responses used by
`plot_supplementary_figure_08_damping_observer.m`. It uses the supplied ten-mode
RT60 vectors and preserves saved choices for exact ties. The decision rule
selects the sound with the shorter mean modal RT60 as damped.

Amplitude and centroid observer scripts and saved results are in
`Figures/Figure 9/`. Saved-output figures can be regenerated; fresh simulations
require the original assembled trial WAV sets, which are not included here.

## Data and experiment samples

### Human experiments

Response paths below are relative to `Human_Experiments/`.

| Experiment | Paper analysis N | Response input |
|---|---:|---|
| 0 | 51 | `Experiment 0/Experiment_0_data.csv` |
| 1 | 87 | `Experiment 1/results_exp1.mat` |
| 2 | 57 | `Experiment 2/Expt2_data.mat` |
| 3a | 101 | `Experiment 3a/Exp3a_data.mat` |
| 3b | 118 | `Experiment 3b/Exp3_data.mat` |
| 3c | 60 | First 60 rows of `Experiment 3c/Exp3c_data.mat` |
| 4 | 55 | `Experiment 4/Exp4_data_pre_rerun_20260809.mat` |

The Experiment 4 paper-response file differs from the first 55 rows of the
later combined response file; the statistics runner selects the original input
explicitly. Later participants are retained for headphone analyses: Experiment
3c has 84 combined participants and Experiment 4 has 194.

[`participant_data/`](participant_data/) contains original, additional, and
combined metadata. The dedicated demographics summary distinguishes `Other`
from `DidNotAnswer`; the older three-category MATLAB summary combines them
under `Other`. Recruitment-platform IDs are omitted from the shared response
and metadata files; numeric within-study identifiers are retained where used.

### Object survey and acoustic recordings

[`data/object_survey/`](data/object_survey/) contains James Traer's anonymized
object-survey responses and material/size summary workbook. Its README describes
provenance and columns.

[`data/README.md`](data/README.md) explains the example impulse responses,
preprocessed observed recordings, modal parameter tables, and mapping files.
`observed_1.wav` files are observed, preprocessed IRs used in the modal analysis;
they are not model-resynthesized waveforms.

**The complete raw IR recording archive, including object photographs, is
distributed separately and is not included in this repository.** Contact the
authors for access while its archival release is being prepared. Bundled
recordings and saved results support the workflows described here.

## Repository layout

| Path | Purpose |
|---|---|
| `Human_Experiments/` | Responses, statistics, plots, and headphone analyses |
| `participant_data/` | Participant metadata and sample splits |
| `stats_results/` | Reference statistics and saved inference results |
| `Figures/` | Main figure scripts, inputs, and reference SVGs |
| `supplementary_figures/` | Supplementary analyses, inputs, and saved outputs |
| `synthesis_model/` | Modal IR and impact-force synthesis |
| `experimental_stimuli/` | Human experiment stimulus generators |
| `mode_inference_code/` | Modal fitting and bundled cochleagram utilities |
| `observer_models/damping/` | Saved observer inputs and deterministic replay |
| `data/` | Acoustic inputs, modal statistics, examples, and object survey |
| `extra_scripts/` | MATLAB filterbank and table/CSV helpers |
| `outputs/` | Newly generated local results; ignored by Git |
| `FILE_MANIFEST.csv` | File sizes, hashes, and source provenance |

## Reproducibility and scope

A MATLAB R2025b run with 10,000 resamples reproduced all 40 supplied statistics
rows exactly. Damping replay reproduced every saved response. The fitting path
fixes affect argument handling, input discovery, and output writing; model
equations, loss, optimizer, and stopping rules are retained. Experiment 3b's
Recorded-condition bootstrap uses its actual participant responses.

Fixed streams make the statistics repeatable. Other stochastic figure,
synthesis, and fitting scripts may vary between runs. Full optimization,
complete stimulus regeneration, and the full Monte Carlo figure suite are
larger workflows than the example checks. Historical fitting and observer
variants remain in reference subdirectories.

## Citation, attribution, and contact

If you use these data or methods, please cite:

> Agarwal, V., Traer, J., Schwartz, J., & McDermott, J. H. (2026).
> Intuitive knowledge of object acoustics enables perceptual separation of
> physical variables from impact sounds. *Proceedings of the National Academy
> of Sciences*. https://doi.org/10.1073/pnas.2605876123

Machine-readable citation metadata are in [`CITATION.cff`](CITATION.cff).

Bundled third-party code retains its original names, attribution, and license
notices. The cochleagram utilities credit their upstream authors; the
A-weighting and ERB conversion helpers retain their source attribution.
No project-wide software license has been added; contact the authors regarding
reuse and distribution terms. The paper's publication license does not replace
bundled software license notices.

For scientific or data questions, contact **Vinayak Agarwal**
([vinayaka@mit.edu](mailto:vinayaka@mit.edu)) or **Josh H. McDermott**
([jhm@mit.edu](mailto:jhm@mit.edu)). For code problems, open an issue with your
MATLAB/Python version, the command you ran, and the complete error message.
