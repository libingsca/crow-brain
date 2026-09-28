# Experiment status — 2026-09-24

**Both branches completed: A stopped on evidence; B stopped with scoped H-B1 supported.**

Six adaptive rounds completed. Thirty-nine gradient-trained runs complete; failed learning runs retained. Fixed random-network controls, calibration and shape intervention complete. The 10 renderer-interrupted final evaluations were uniformly recovered from unchanged saved weights; original failures retained. Nine tests pass.

- H1 limited phenomenon sufficiency: supported in specified end-to-end synthetic setting; not universal across implementations.
- Existing-model useful shape transfer: excluded by Round 5's predeclared criterion.
- H2-core useful E/I repair with shared frozen encoder: excluded; paired primary effect practically equivalent within ±.05.
- Real NCL causal mechanisms, attractors and connectome-derived primitives: untested, not falsified.

Main report: reports/FINAL_REPORT.md.
Stopping decision: docs/decisions/009_stop_current_branch.md.
No training or monitoring process is left running. A future research restart needs a new identifiable question or new evidence, not more motif search on this same renderer.

Branch B: two adaptive rounds complete. B1 remains quality_gate_failed because a single shuffled-label control, selected using true validation labels, exceeded its registered gate. B2 froze the same heads, confirmed on new images, and passed repeated-null, reset and state-shuffle controls. Old/new 15-step BA: 0.48365/0.97930. Core and head artifacts independently audited. See [B final report](../reports/branch_b/FINAL_REPORT.md) and [B003 stop decision](decisions/branch_b/003_stop_supported.md). Noise robustness is heterogeneous and not a general conclusion. No experimental or monitoring processes remain.
