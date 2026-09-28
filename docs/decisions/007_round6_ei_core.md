# Round 6 — minimal E/I core comparison after Round 5

## Hypothesis and scope
Round 5 shows shape transfer failure while circle performance is saturated; fixed random features do not solve the shifted task. Test H2-core: can an 80/20 Dale-constrained recurrent core repair useful cross-shape generalization WITHOUT new visual learning? This is conditional on this task and common encoder, not a biological NCL fit or a claim about all E/I networks.

Shared encoder from Round 3 DEVELOPMENT seed 1, lr=.001 final checkpoint; no final-model/test weights used. Freeze CNN/project/normalization affine weights. Calibrate normalization using each run's unlabelled training image bank, then freeze statistics. Both arms use exactly the same initial EFFECTIVE recurrent matrix (Dale-compatible, positive/negative magnitudes balanced in expectation by the 80/20 ratio (individual row sums are not forced to zero), spectral radius .9), input weights and readout. Unconstrained arm can subsequently change signs; E/I arm preserves output signs. Initial matrices need not be the same as previous ordinary RNN studies; both arms are retrained for this comparison.

Cache fixed encoder output for speed; verify cached forward agrees with pixel forward. Task, loss, hidden size, alpha, image budget and updates otherwise unchanged. Each arm gets 3 development seeds × lr .001/.003 × 1500 updates, select mean last ID validation BA, ties lower lr. Independently select learning rate for each arm. Final seeds 301–305, 2000 updates; no test-dependent extension or seed replacement. New final image streams 220000+; shape streams 230000+. Confirm normal-arm ID lower seed-CI >.75 before interpreting structural results; retain failed E/I runs as part of the fixed-budget comparison, without generalizing to unlimited tuning.

## Endpoints and interpretation
Primary: mean solid→ring and ring→solid BA. Useful repair requires E/I lower 95% seed-t bound >.65 AND paired E/I-minus-unconstrained lower bound >.05. Exclude the specified useful-repair hypothesis if the E/I upper bound <.65; exclude a ≥.05 relative benefit if paired upper bound <.05. Equivalence of the paired difference only if its entire interval lies inside ±.05. Other cases inconclusive, not null acceptance.

Secondary: ID/area-shift performance, reset, 15-step delay, independent quantity decoding and peaked tuning. Long-delay comparisons are exploratory because shape transfer is primary. Primary conclusion concerns learned core given a frozen encoder, and cannot establish that E/I with a differently learned frontend never helps.

## Resources and decision
Two CPU processes, one torch thread each, 900 s/run cap; 12 development and 10 final runs maximum. No hidden post-test tuning. If useful repair excluded and remaining advantages are limited to already-solvable circles, further motif/architecture expansion on this renderer has low information value; stop that branch with a concrete report, not a universal rejection of avian mechanisms. If useful repair supported, design a new intervention from the result. If primary inconclusive, assess uncertainty and alternative explanations before deciding the next independent experiment.
