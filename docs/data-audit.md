# Data access audit — 2026-09-24

## Ditz et al. 2022

Dataset [10.17632/7kmtz7kj4c.1](https://data.mendeley.com/datasets/7kmtz7kj4c/1), creator Andreas Nieder, CC BY 4.0 (verified in dataset metadata). Downloaded all 16 listed files with SHA-256 checked against provider metadata. Exact names, sizes, hashes and MATLAB variables: `data/evidence/download_manifest.json`. Original files in `data/ditz2022/`; no modification to originals. Any derived plots will be labeled as our reanalysis.

The first guessed API path returned 404; the published page's own JS revealed the working public file-list endpoint. Python urllib received 403 for a download; curl succeeded. Larger files needed timeout retries. These were access-tool failures, not evidence that data were unavailable.

Inspected variables and supplied plotting code:

- `data_behavior.mat`: aggregate behavior curves for small and large protocols.
- `data_firingrate.mat`: arrays for two example recordings (564×2700), per-unit count-conditioned activity arrays for 1136 units and selection/preference fields.
- `data_physio.mat`: per-unit aggregate time courses and spike-density arrays.
- `data_tuningcurve.mat`: selected-unit aligned tuning matrices, separated into putative BS/NS and sample/delay, small/large protocols.
- `data_tempcorr.mat`, `data_tuningadjacent.mat`: processed pair summaries.
- supplied MATLAB scripts explain figure construction, not a complete trial-level analysis pipeline for all units.

**Usability**: processed unit-level tuning data are available; it would be incorrect to claim that only paper text is accessible. Full raw trial metadata, complete stimulus images, animal/session mapping and repeated measurements for every unit have not been identified in these files. Do not treat selected-unit matrices as independent animal replications. Two example trial recordings do not supply all-unit trial-level cross-validation.

**Current experiment boundary**: our 2–6 synthetic count set and controlled extrema geometry differ from the biological protocols. These data permit reference plots and selected aggregate comparisons, but the current model is not a preregistered quantitative replication of NCL. H2 biological fit requires a separately specified matched protocol and observation model; final simulation decoding is not a substitute.

## Other sources

2016 protocol details checked against PubMed's original article abstract and figure captions; complete raw data not acquired. Stacho 2020 / Area X are background for boundaries only. The connectome branch is not running and its API/data availability has not been audited in this execution.
