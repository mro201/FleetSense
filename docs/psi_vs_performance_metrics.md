# Design Note: Feature Drift (PSI) vs. Model Performance

## What we found

Feature-level PSI shows a broad, multi-class seasonal pattern, strongest in
autumn through winter, concentrated in positional features (`lat_mean`,
`lon_mean`), speed percentiles, and reporting frequency (`n_pings`,
`time_span_seconds`). Fishing is affected most severely, but Passenger,
Tanker, and Tug all show the same shape at the same time, and Cargo shows a
weaker version of it too.

Despite that, per-class F1 stays essentially flat across the same period, and
the predicted-class balance (`__predicted_class_balance__`) barely moves at
all over the whole year.

## Why these don't contradict each other

The features driving the PSI spike (`lat_mean`, `lon_mean`) ranked low in
permutation importance. The model was never relying on them much to begin
with. So a real, substantial shift in the input distribution can happen
without moving accuracy or predictions, because the model isn't using that
signal to decide.

This is the intended value of running feature-level and prediction-level
monitoring together, rather than either alone: feature PSI alone would have
looked alarming here; performance/prediction monitoring alone would have
missed the shift entirely. Together they show the fuller picture: the world
changed, the model didn't need to notice, and we can say why.

## Caveats, not yet resolved

- Only about one year of data — "seasonal" is a plausible read, not a
  confirmed one. Need a second winter to actually distinguish "recurring
  pattern" from "one unusual season."
- Tug's `time_span_seconds` spike in December is a single very dark cell,
  isolated from its neighbors — looks more like a low-sample-size artifact
  (Tug is a low-volume class) than genuine drift. Worth checking actual
  monthly counts for Tug before folding it into the seasonal story.
- Cargo's F1 shows a small but real downward wobble at the very end of the
  period, and it's the only class not flagged by PSI at all. Either normal
  noise, or a subtler drift that hasn't crossed the threshold yet.

## Next step: weighting drift by permutation importance

Right now every flagged feature is treated as equally alarming regardless of
whether the model relies on it. The natural next step is to combine PSI with
the permutation importance already computed, so drift in a feature the model
actually depends on gets prioritized over drift in one it barely uses.

Two things to build, not yet started:
1. A weighted drift score per period (PSI weighted by importance, summed or
   averaged across features) — a single number per period reflecting how much
   the drift actually matters to the model, not just how much the inputs
   moved.
2. A re-ranking of the existing flagged-breach list by `psi * importance`, so
   the most consequential breaches surface first without hiding any
   individual alarm (unlike a single aggregated score, which can dilute a
   severe breach in one important feature among many stable ones).

Caveat to keep in mind when building this: permutation importance was
computed globally, not per class — the weighting will be a reasonable proxy,
not an exact per-class picture, since a feature's importance may differ
across classes in ways the current importance numbers don't capture.
