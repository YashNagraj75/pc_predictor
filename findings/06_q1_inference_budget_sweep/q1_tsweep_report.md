# T sweep: more inference makes the encoder worse

Same code, same 400-epoch budget (17,200 steps), same everything as the `_v2fix`/`_e400`
grid. Only `cfg.pc.T` changed. Runs tagged `_Tsweep`. All three cells below completed the
full budget (final step 17,199).

| T | invariance loss at feedforward embeddings | invariance loss at relaxed activities | energy F | kNN probe | effective rank | TwoNN | dims to 95% |
|---|---|---|---|---|---|---|---|
| 16 | 0.6942 | 0.5744 | 0.0018 | 0.3305 | 20.81 | 22.71 | 50 |
| 32 | 0.7469 | 0.2544 | 0.0073 | 0.2970 | 7.33 | 19.84 | 29 |
| 64 | 0.9109 | 0.0371 | 0.0075 | 0.2655 | 1.71 | 11.31 | 2 |
| backprop | 0.3999 | n/a | n/a | 0.3910 | 18.33 | 9.22 | 20 |

Random-init reference: kNN probe 0.343. Best hidden-layer probe is 0.3935 at every T.

T=128 was killed at step 2,580 of 17,200 on the user's instruction once the trend was
established; T=256 never started. The partial T=128 run directory is left on the host and
its four evaluation points are in `tsweep.json`, but they are not comparable to the rows
above and are excluded from the table.

## The finding

Every quantity is monotone in T, and they move in opposite directions:

- The objective **at the relaxed activities** falls 15x, 0.5744 -> 0.0371. More relaxation
  steps solve the task better and better.
- The objective **at the feedforward embeddings** *rises*, 0.6942 -> 0.9109.
- The probe falls 0.3305 -> 0.2655, from just below the random-init line to well below it.
- The embedding collapses: effective rank 20.81 -> 1.71, dims-to-95% 50 -> 2.

`evaluate()` in `train.py` extracts embeddings with `E.forward` for both arms (the function's
own comment states this), so the probe measures the feedforward map. That is also how a
joint-embedding encoder is deployed: one forward pass per image.

## Mechanism

The activity relaxation and the feedforward weights are two independent routes to satisfying
the objective, and they compete for it. Given more relaxation steps, the activities absorb
more of the objective themselves; the residual that drives the weight update shrinks, so the
weights are never pressured into a feedforward map that is good on its own. In the limit the
relaxation does the work and the feedforward pass drifts — here, toward collapse, because the
distributional regulariser's pressure is absorbed by the activities too.

## What this corrects

The earlier `_e400` report read PC's higher feedforward loss (0.6942 vs backprop's 0.3999) as
**under-optimisation**, and proposed raising T as the fix. That diagnosis was wrong. Raising T
does fix the relaxed-activity loss and makes the deployed loss and the probe strictly worse.
Two further signs the under-optimisation reading was never supported: the energy was already
converged at T=16 (F = 0.0018, and it does not improve with more steps — 0.0075 at T=64), and
the best hidden-layer probe is pinned at 0.3935 across all three T values, i.e. unchanged.

## Consequence for the research question

What this grid tests is **PC for the encoder**. An encoder is consumed feedforward, so the
decoupling above is a direct penalty on the quantity that matters. The original proposal's
scope was **PC for the predictor** inside a joint-embedding world model, where the output is
consumed as a prediction rather than as a deployed embedding — that configuration is not
argued against by this result. The glimpse/saccade frontend changes the input distribution and
so cannot address a decoupling that lives in the learning rule; it remains worth running as a
validation control that the pipeline can reach published-looking numbers.

## Not assessed

- One seed per cell; no seed variance.
- T below 16 (T = 4, 8) — would test whether the trend is monotone all the way down.
- Whether the decoupling is specific to the coupled `output_drive` setting.
- Whether reading the probe off the *relaxed* activities recovers accuracy. This is the
  obvious follow-up, but it is not the deployment condition for an encoder.
