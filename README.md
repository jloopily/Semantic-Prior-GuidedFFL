# Semantic-Prior-GuidedFFL

Research code for **Semantic-Prior-Guided Face Forgery Localization with Real-Image References**.

The current experiment, **R1**, asks whether semantic relation scores can localize Face2Face manipulation using a separate real-image gallery instead of the target video's paired real frame. The project uses frozen DINOv3 patch features and independently detected facial regions.

**Status:** R1 implementation and test cases have been written, but have not been executed or validated in the local editing environment. This repository contains no R1 performance results or publication acceptance claims. The title is a working research title.

## Current R1 implementation

- Train-only source-video-component separation between gallery, calibration and evaluation.
- Independent face crops and MediaPipe geometry for every query and reference.
- Frozen DINOv3 ViT-B/16 block-6 features, 448-pixel input and a 28×28 token grid.
- Four registered methods: `relation`, `local_difference`, `position_template` and `broken_context`.
- Fixed real-only gallery seeds and equal-video aggregation of pairwise distances.
- Clean-real c23 threshold calibration and unchanged threshold transfer to c40.
- Pixel AP, F1/FPR, semantic-region diagnostics and connected-component bootstrap.
- Mask access only during evaluation, after score maps have been committed.
- Default evaluable-frame coverage gate: 50%.
- Resumable stages, provenance receipts and resource checks.

The main pipeline is:

```text
prepare → crops → geometry → features → gallery → calibrate → score → evaluate → finalize
```

Entry point: [`src/run_r1.py`](src/run_r1.py). Configuration: [`configs/r1_non_oracle_semantic_relation_v1.yaml`](configs/r1_non_oracle_semantic_relation_v1.yaml). Full implementation details and stage commands: [`R1_LOCAL_IMPLEMENTATION.md`](R1_LOCAL_IMPLEMENTATION.md).

## Repository contents

| Path | Purpose |
|---|---|
| `src/r1_non_oracle_semantic_relation/` | Current non-oracle reference experiment |
| `src/r0_semantic_relation/` | Historical paired-real diagnostic and reusable mathematical routines |
| `src/cirlid/` | Shared data, feature, geometry and evaluation infrastructure; earlier experimental methods |
| `configs/` | Current and historical experiment configurations |
| `tests/` | Geometry, manifests and R0/R1 test cases |
| `experiment-record/` | Local research records, including negative results and limitations |
| `tools/` | Selected download and environment-preparation utilities |

The earlier CIR-LID name remains in historical modules and documents to preserve their provenance. Those experiments and R1 use different protocols and should not be compared as a single benchmark table.

## Run on an experiment machine

Use Python 3.10+ with NumPy 1.22+, PyYAML, compatible PyTorch/torchvision, OpenCV, dlib and MediaPipe Tasks. Tests additionally require pytest. Obtain the official DINOv3 implementation/weights and MediaPipe Face Landmarker asset separately. Dependency combinations have not been runtime-validated for R1.

FaceForensics++ data must be obtained through its authorized distribution. Configure the dataset, `train.json`, model repository, weights and asset paths before the first run. Relative configuration paths are resolved from the configuration file's directory.

From the repository root on a suitably provisioned experiment machine:

```bash
python -m pytest tests/test_r1_non_oracle_semantic_relation.py -q
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage all --allow-heavy
```

To resume the same frozen run:

```bash
python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage all --resume --allow-heavy
```

R1 defaults to CUDA feature extraction. The full runner rejects CPU quotas below 2 cores or memory limits below 4 GiB; these are minimum refusal thresholds, not measured capacity recommendations. Runtime and GPU-memory requirements have not been benchmarked. Do not run this pipeline on a 0.5-core / 2-GiB host.

## Research interpretation

Earlier local records document unsuccessful or confounded localization hypotheses as well as positive structural observations. R0 showed an advantage over unrestricted local feature differences, but did not establish an advantage over its semantic template or broken-context controls.

A support-matched local baseline, applying the same mouth/perioral restriction as `relation`, has been proposed as a further diagnostic. **It is not implemented in this snapshot.** A pipeline decision such as `PASS_CONTINUE` is an internal research rule, not a journal acceptance criterion.

Official test access remains locked for the current R1 pilot. Source-video-component separation does not establish biometric identity separation. Independent crop-grid correspondence does not imply landmark registration.

## Published scope

This repository contains selected files from the local project. Server report bundles, server connection scripts/passwords, raw videos, model weights, feature arrays, generated results and personal documents are excluded. Some historical records reference local paths or external result bundles that are not included; those records do not make this a complete reproduction archive.

The FaceForensics download utilities originate from the FaceForensics project and retain their upstream references. External datasets, model implementations and assets remain subject to their respective access conditions and licenses. No new license has been assigned to the repository in this initial upload.
