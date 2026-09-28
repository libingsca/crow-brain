# Round 4 — inference-statistics diagnosis after completed Round 3

Round 3 completed all 5 final seeds. Four learned well; seed 103 had low training loss but ID .6225 / shift .540625 at final inference. Other seeds ID 1.0. This motivates a narrowly specified diagnostic, not replacing or deleting the failed run.

Change only BatchNorm running mean/variance: recompute exactly from each model's original training image bank (no labels, validation or test inputs), with all learned parameters frozen. Apply to all five seeds, not just seed 103. Evaluate original and recalibrated models on a NEW common test stream 140000–140002, 4000 balanced trials per condition; analyze representation on independent 140010+ streams. Save both results and calibrated state.

Operational diagnosis: if all seeds recover ID ≥.90 with unchanged learned parameters, inference-statistics mismatch is a sufficient explanation for this observed failure. This is an adaptive diagnostic on previously seen model seeds, not a new independent training replication. A later architecture comparison must use fresh seeds and apply identical calibration to both arms.

Do not choose calibration strength using test data: use exact unweighted training-image population mean and unbiased variance, preserving BatchNorm epsilon and affine parameters. No new training. Preserve Round 3 original results. E/I experiment remains conditional on this result.
