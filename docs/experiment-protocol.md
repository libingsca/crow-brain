# Round 1 protocol — registered before training, 2026-09-24

## Scope and provenance
Synthetic visual delayed match, counts 1–5, motivated by small-number protocols pooled in Ditz et al. 2022. Not an exact reproduction of the 2016 task (counts 1,2,4,7,12,20,30; 800 ms sample/1000 ms delay). Abstract simulation steps have no fitted biological time constant. Only the first match/nonmatch decision is modeled. No raw NCL trials have yet been downloaded; biological-fit H2 is not testable at this stage. The primary neural proxy is held-out five-way late-delay decoding, not fit to NCL.

## Data and controls
32×32 grayscale disk arrays, nonoverlapping dots with randomized centers. Independent sample/test images. Train mixture: standard independent radius 1.5–2.5 pixels; area-control common radius sqrt(U[7,11]/count). Discrete raster area is recorded, not assumed exactly equal. Independent nuisance parameters for each image. Cue-shift test uses radius sqrt(U[9,13])/count**0.65, reversing the total-area/count trend, and variable spatial extent. No explicit count is provided to the model, no auxiliary count loss.

Counts uniform; match probability 1/2 at every sample and test count. Nonmatch count selected uniformly from the other four. Evaluation uses exact stratification of ordered count pairs and labels. Training/test/analysis image RNG streams are disjoint. No-memory reset at test onset and test-only classifier check for leakage. Classifiers on measured image nuisance statistics audit remaining shortcuts rather than assert perfect cue independence.

## Model and optimization
Shared visual CNN (two small convolution layers + spatial sum + linear projection), rate RNN with 64 nonnegative ReLU states, leak alpha=0.5, 4 sample steps, 5 blank delay steps, 4 test steps, linear binary decision head from final state only. Blank input zeroed after encoding. Ordinary signed recurrent matrix; Dale-constrained variant is deferred to a subsequent round designed from Round 1. End-to-end encoder learning is part of the ordinary model, not independently attributed to recurrence.

Round 1 E0 includes untrained checkpoints, fixed random core + fixed encoder + trainable readout, and fully trainable ordinary RNN. Reservoir is an ablation, not an optimized reservoir-family benchmark.

Development seeds 1,2,3, learning rates 0.001 and 0.003, at most 1500 updates each, batch 64. Select learning rate by mean ID validation accuracy, tie by lower loss. Final runs: 5 fresh seeds 101–105, 2000 updates each, no adaptive extension after looking at test results. Initial/final checkpoints saved. Validation selects hyperparameters, not final test checkpoints. No silent dropping of failed seeds. Controls receive identical sample budgets and learning-rate candidate counts.

## Metrics and decision
Primary behavioral metric: balanced accuracy on new cue-shift images. Primary representation proxy: ridge five-way decoding of sample count from late-delay state, fitted and tested on independent balanced images from cue-shift protocol. ID accuracy and reset accuracy are gates. Secondary: encoder decoding; selection on ID analysis trials and independent tuning on cue-shift trials; interior-peaked units; extended-delay accuracy.

Model seed is the replication unit. Report all five seeds, mean and two-sided 95% Student-t CI (descriptive with small n); no neuron-level pseudoreplication. H1a limited functional support requires lower CI of ID task accuracy >0.75, cue-shift accuracy >0.65, delay decode >0.40 (chance 0.20), and reset accuracy within 0.45–0.55 per seed. These are operational functional thresholds, not biological fidelity claims. H1b interior-peaked tuning is assessed separately: significant count effect with BH FDR .05 on selection trials, interior maximum, and same preferred count with positive prominence against both endpoints on independent cue-shift trials; lower CI of fraction >0.05 supports reproducible peaked units. Counts 1–5 cannot test broad-range Weber scaling.

If gates fail, diagnose before structural inference. If H1a holds but H1b does not, do not call all H1 supported. E/I comparison and its δ (provisionally 0.03 BA and 0.05 decode) will be fixed in a NEW protocol only after this round's report. Subsequent rounds require an explicit link to the preceding results and new final-test RNG streams.

## Resources and stopping
2 CPU cores, ~3.3 GiB memory, no GPU; use 1 torch thread to avoid oversubscription. Round 1 budget: 4 wall hours maximum, each run at most 15 minutes, no paid cloud services. This is an execution budget, not evidence that research value is low. If performance fails, record and design the next diagnostic round instead of extending runs until a desired outcome appears. Overall stopping requires scoped hypothesis resolution or a documented low-information-value decision; budget alone is not a scientific stop.
