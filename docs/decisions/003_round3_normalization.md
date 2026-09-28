# Round 3 pre-registration — driven by Round 2

Observation: all six runs at chance despite weakly decodable states. Intervention: add feature-wise BatchNorm after the visual projection; no change to recurrent weights, task, target, loss, training grid or time discretization. This is a numerical optimization diagnostic, not a claim about avian circuitry.

Evaluation MUST call eval() and freeze normalization statistics. Models never access trial labels during encoding. Freeze normalization buffers for reservoir. Keep tests for reset independence and ensure single-trial inference agrees with batch inference.

Development: same 3 seeds × 2 learning rates × 1500 updates. If mean ID validation BA >.75 for a candidate, select highest mean final validation BA (tie lower numeric lr); then run exactly 5 new seeds 101–105 at 2000 updates with test base 110000. No test-dependent extension. Same H1a/H1b thresholds. Untrained and trained representation checkpoints saved. Reservoir comparison, if baseline succeeds, receives the same lr grid and sample budget. If development fails, diagnose again based on completed round.

BatchNorm is an artificial frontend normalization, not a fitted NCL feature. E/I comparison, if warranted later, must share this encoder.
