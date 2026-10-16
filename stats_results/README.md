# Supplied reference statistics

These are the final September 6, 2026 statistics collected from the working
code folder. `supplementary_statistics.csv` contains 40 comparisons, and the
MAT files contain detailed results and resampling draws.

Paper analyses use N = 60 for Experiment 3c and the original N = 55 response
file for Experiment 4. Later participants are retained for headphone analyses
and are not pooled into these paper tests. The Experiment 3c performance
differences include 95% participant-bootstrap confidence intervals.

`Exp3c_Exp4_demographics_for_Josh.csv` reports original, additional headphone-test,
and combined demographics, with did-not-answer responses separated from other
gender responses, plus mean age and age SD.

Regenerate statistics with `run_paper_statistics(10000)` and demographics with
`export_exp3c_exp4_demographics`. Both accept another output directory.
The package README defines the methods and the intentional threshold bounds.
