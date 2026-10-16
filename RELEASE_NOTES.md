# Release notes

## Repository preparation

- Added the paper title, complete author list, abstract, citation metadata,
  setup instructions, workflow examples, and data/output documentation.
- Repaired modal-fitting argument handling and output paths. The example now
  accepts a WAV or directory through `--input`, provides usable defaults,
  reports its output location, and creates checkpoint folders before writing.
  Model equations, loss, optimizer, and stopping rules are retained.
- Included James Traer's anonymized object-survey CSV and summary workbook.
- Applied the approved Experiment 3b correction: Recorded bootstraps use actual
  responses rather than a perturbed copy of Full. The duplicate entry point
  uses the included input filename and creates its output directory.
- Retained statistical calculations, reference statistics, metadata,
  third-party files, and other figure error-bar conventions.

## Analysis provenance

The release includes original paper responses and revised participant-level
statistical analyses. Experiment 3c uses 60 original participants; Experiment 4
uses the original 55-participant file. Additional participants remain available
for headphone analyses. Project scripts have descriptive names and updated
references. `FILE_MANIFEST.csv` records source provenance, sizes, and hashes.
The larger research archive and full raw IR corpus are separate.

The portable damping observer replays 200 × 525 saved trials using the exact
RT60 vectors and saved tie choices. Historical assembly scripts are retained.

## Validation and scope

The reference MATLAB R2025b run with 10,000 resamples reproduced all 40
statistics rows exactly. Damping replay reproduced all 105,000 saved responses.
Repository preparation checks cover the repaired fitting entry point and
output writing, syntax, hashes, and the approved Experiment 3b correction.
Full optimization, fresh stimulus generation, and the complete Monte Carlo
figure suite are larger workflows and are not implied by these checks.
