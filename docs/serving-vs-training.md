# What Gets Frozen at Training Time, and Why Serving Is Separate From Training

Every training run in this project produces a set of artifacts that are saved once and then reused, unchanged, until the next retrain. Serving never recomputes any of them live. This note explains what those artifacts are, why serving is built to depend on a frozen snapshot rather than recomputing anything fresh, and a real gap in the current implementation that this way of thinking exposed.

## What training produces and freezes

A training run saves four things.

The **model artifact** is the trained RandomForest classifier itself. It is written to disk once per training run and loaded as-is by the serving process from then on; the API never retrains as a side effect of answering a prediction request.

**Training metadata** records the specifics of that run: the training window, the data used, and related bookkeeping. This is what lets a saved model be tied back to exactly how it came to exist, rather than being an anonymous file.

The **PSI baseline** is the set of reference bin edges and proportions computed once from the training window's data. Every later drift comparison is measured against this same fixed baseline rather than against a baseline recomputed on the fly, so that a PSI value from one period means the same thing as a PSI value from another.

**Feature importance** (permutation importance, used as drift weights) is also computed at training time and saved alongside the baseline. This is deliberately not recomputed on every monitoring pass — it's an expensive computation, and it only needs to reflect what the current model actually cares about, which only changes when the model itself changes.

Separately from all of this sits the feature schema. Unlike the four artifacts above, the schema is not something training produces. It is a fixed structure that already exists before a training run starts, and training is expected to conform to it, not the other way around. The schema is the one thing both training and serving share and depend on equally; neither side owns it or is free to change it unilaterally.

## Why serving doesn't recompute anything live

An alternative design would be for the API to import the training code directly and retrain (or at least recompute) on every request, so that training and serving are always trivially in sync — there would be no separate artifact to go stale in the first place.

This doesn't work in practice, for three reasons that show up directly in this system.

A single training run takes a few seconds; a single prediction needs to return in milliseconds. If serving triggered any part of that live, every prediction would inherit training's latency, which is not acceptable for something meant to answer in real time.

Serving a fixed, frozen model also guarantees that identical inputs produce identical outputs. If two people submitted the exact same vessel features one after another, and a live retrain happened to complete in the moment between their two requests, the second person could get a different answer than the first — despite asking the identical question. Freezing the model at training time and serving that same fixed artifact until the next explicit retrain removes that possibility entirely.

Finally, training and serving need to be able to fail independently. A slow, memory-heavy, or broken training run should never be able to block or take down prediction traffic. Keeping them as separate processes, connected only through the saved artifacts, is what makes that isolation possible.

## The gap this exposes: schema drift and silent wrong predictions

Thinking through why these artifacts are frozen surfaces a sharper question: what happens if the schema itself changes after a model has already been trained against it?

A RandomForest classifier has no runtime concept of feature names. At prediction time it operates on a plain ordered array of numbers; it trusts that position N in that array means the same feature it was trained on. If the schema were changed after training — a feature reordered, or a new one inserted before an existing one — the position a trained model expects for a given feature would silently shift underneath it.

Critically, this would not raise an error. The model would still run, still return a prediction and a full probability distribution, exactly as if nothing were wrong. It would simply be scoring the wrong feature under the wrong name, with complete confidence. A silent wrong prediction is a worse failure than a crash, because nothing downstream would have any reason to suspect a problem.

As it stands today, nothing in the pipeline checks for this. The saved model artifact and the schema file on disk are only ever implicitly assumed to match.

The fix follows directly from how the other frozen artifacts already work: store the exact feature list — names, in order — inside the training metadata at the moment a model is saved, so every artifact carries a permanent record of the schema it was actually trained against. Then, at serving startup, compare that stored feature list against the schema file currently on disk. If they don't match, the server should fail to start, rather than come up and quietly serve predictions against a contract that no longer holds.

That is really the same idea as everything above it, seen from a different angle: the frozen artifacts and the separation between training and serving both exist to stop training and serving from drifting apart mid-flight. The one place they could still drift apart silently — a schema change slipping past unnoticed — is exactly where a check needs to exist, and currently doesn't.
