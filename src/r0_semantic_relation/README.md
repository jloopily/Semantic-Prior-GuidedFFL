# R0 semantic-relation oracle diagnostic

This is an isolated, train-only Face2Face/c23 experiment. It uses a strictly
same-frame paired target-real image to define semantic coordinates and applies
the same coordinates to the fake image. The oracle dependency is intentional
and means the result is **not a deployable detector**.

## Stages

```text
python -m src.run_r0 --config configs/r0_semantic_relation_v1.yaml --stage prepare
python -m src.run_r0 --config configs/r0_semantic_relation_v1.yaml --stage crops
python -m src.run_r0 --config configs/r0_semantic_relation_v1.yaml --stage features
python -m src.run_r0 --config configs/r0_semantic_relation_v1.yaml --stage evaluate
python -m src.run_r0 --config configs/r0_semantic_relation_v1.yaml --stage finalize
```

`prepare` opens only `train.json`; validation/test paths are kept in the
configuration for provenance but are not opened. The official test remains
locked. Canonical undirected target/source pairs are recorded as source groups, so reverse
directions share a provenance label. For the inferential bootstrap, the runner uses the
more conservative train-split source-video connected component as the resampling unit;
this prevents videos linked through shared source/target identities from being treated
as independent. The report shows both counts. The train split is divided by source-video
connected component into deterministic exploration and holdout partitions;
the final decision uses holdout only.

Each later stage rejects a missing or changed freeze receipt, algorithm source hashes, DINOv3 checkpoint hash, or MediaPipe asset hash, and `prepare` will not overwrite an existing frozen manifest. A pair is evaluable only when the fake frame index, target-real frame index,
crop-source frame index, and mask frame index satisfy the frozen strict pairing
rule. Mismatches remain in `pairing_manifest.jsonl` and are reported rather
than silently repaired or dropped.

`crops` reuses the project's FF++/dlib crop implementation, and `features`
reuses its official DINOv3 block-6 interface. No gallery or token kNN is used.

The primary relation score constructs a mouth-to-left/right-cheek relation
vector separately for fake and target-real features, then measures their
per-token change. The semantic coordinates are still defined from target-real
landmarks only. The comparators are a semantic position template, local paired
DINO difference, and a fixed context permutation that changes cheek context
only while preserving mouth/query token positions.

The secondary mouth/perioral-conditioned evaluation is planned, not implemented in R0. Pixel AP groups tied score values at one threshold, making binary templates and
constant heatmaps invariant to arbitrary within-tie ordering. Every scorer
is evaluated on the same full-face valid pixel domain (relation and template
remain zero outside their predefined mouth/perioral support). The paired
bootstrap resamples source-video components with replacement and reports its
estimand and group sizes.

The evaluator also writes target-real and fake-only region overlays, fake-only geometry IoU diagnostics, mask coverage and empty/full valid-mask audits. All artifacts and pair-level exclusions are written below
`results/r0_semantic_relation_v1`, including `quality_audit.json`,
`per_frame_metrics.jsonl`, `per_video_metrics.jsonl`, partition summaries,
and `decision.json`. The decision never unlocks official test. A positive decision requires at least five independent holdout components, the predeclared 90% strict-pair and single-target-real-face quality minima, an 80% evaluable-pair minimum (globally and in holdout), no more than 5% unexpected evaluation errors, and positive clustered CIs against both main comparators and the context-only shuffle. Inadequate quality or independence yields an inconclusive result rather than a negative scientific claim. Matching decoded frame indices and crop coordinates is a conservative audit, not independent proof that independently encoded videos have identical timestamps; visual pairing checks remain necessary.

The MediaPipe Tasks asset must be an official `face_landmarker.task` file. Set
its SHA-256 in the frozen configuration after placing it on the server; do not
silently use a different model.

Video-level pooled AP uses all three geometry-valid frames, including all-zero-mask frames whose individual AP is undefined. Videos with fewer than three geometry-valid frames are ineligible; frame-AP coverage and video coverage are reported separately. Existing freeze receipts produced by older source files must not be reused after this revision: start a new output directory rather than silently rewriting an old frozen run.
