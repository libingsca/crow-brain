# Round 5 — controls driven by Round 4 and the Round 3 boundary audit

## A. Shape transfer on the five frozen calibrated models

Reason: short-delay performance recovered, but boundary statistics explain the circle task almost perfectly. Train nothing. New test image streams 180000+. Compare solid→solid, ring→ring, solid→ring, ring→solid at delay 5, with labels balanced independently of shape. Rings keep one 8-connected object per item; inner radius=.45 outer radius. Use numerical rendering to match total foreground mass and bounding extent distributions to the original mixed protocol. Positions remain random with two extrema anchors; no exact pixel-level counterpart claim.

Primary: average BA of the two cross-shape conditions per model. Operational claim under test: these learned models transfer quantity matching across these shapes with BA ≥.65. Upper 95% seed-t bound below .65 rules out this specified useful-transfer claim; lower bound above .65 supports it. Other intervals inconclusive. Evaluate all models and report all conditions, not only failures. This new condition was not used for training or hyperparameter selection. The seed interval is conditional on the existing model family, not an animal claim.

Validate 8-connected component count for every generated image before evaluation. Failure of this image check stops shape inference pending repair. The transformation is intentionally out of training distribution, so failure limits generalization; it does not negate within-circle H1b or identify the exact internal algorithm.

## B. Fixed random network readout control

Reason: untrained networks already have high quantity decodability; original low-amplitude Adam training would be a weak reservoir control. Freeze random encoder and recurrent core; calibrate encoder normalization on unlabelled training images; fit a standardized ridge linear readout to final state (no count labels, only match labels). This directly addresses numerical readout conditioning. Compare alpha .1/1/10 on development seeds 1–3, 8000 balanced training pairs and independent 1600 validation pairs. Pick highest mean validation BA, ties stronger regularization. Final seeds 201–205, same 8000 pair budget, fresh ID/shift/reset test streams 190000+ and 4000 pairs. Save features, labels, weights, readout and seeds.

This control has a different readout optimizer and smaller task-sample budget than end-to-end RNN; do not use it for a matched sample-efficiency claim or to reject the whole reservoir family. It tests whether fitted random features are already sufficient for the current task. It does not replace the main E/I comparison. Next round requires this round's report and an explicit expected-information-value assessment.
