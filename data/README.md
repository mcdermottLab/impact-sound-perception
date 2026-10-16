# Acoustic data and supporting tables

| Path | Contents |
|---|---|
| `example_IRs/` | Example measured impulse responses (`h.wav`) used by the fitting demo |
| `15_modes_v5/observed/` | Numbered directories of preprocessed observed IRs (`observed_1.wav`) used in the modal analysis |
| `15_modes_v5/` CSVs | Saved modal parameters and original recording/path labels |
| `recorded_impacts/` | Recorded impacts used by Experiment 0 stimulus generation |
| `object_survey/` | Anonymized object survey and material/size summary |

`observed_1.wav` files are preprocessed **observed recordings**, not synthesized
model outputs. They are normalized/truncated targets saved by the fitting
workflow. A numbered directory identifies a recording entry, not necessarily
a unique physical object; objects may have multiple recording positions or
channels.

`dirnames.csv` and `filenames.csv` retain original recording labels;
`directories.csv` retains paths used when assembling modal parameter tables.
Historical source paths document provenance and need not exist on a new
machine. Mapping and parameter tables were assembled at different stages and
should not be joined solely by matching row counts. Included analysis scripts
specify how they use these files.

The complete, unprocessed IR corpus and object photographs are distributed
separately. They are not replaced by these preprocessed analysis inputs.
Contact the authors for the full archive while its archival release is prepared.
