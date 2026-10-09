# CIR-LID v5.5 source

This directory contains the local implementation for the frozen CIR-LID v5.5 protocol.
The code is intentionally split into two boundaries:

## Implemented local stages

- FF++ official-split parsing and explicit c23/c40 + manipulation path enumeration;
- video probing, RGB frame decoding, raw mask decoding, and normalized mask-frame remapping;
- immutable sample manifest generation with split/config hashes, a separate exact-middle-frame gallery manifest, fixed gallery IDs, and fixed validation pilot/reference names;
- dlib HOG target-original face detection, shared square crop boxes, reflection padding, crop/mask caching, and exclusion logging;
- exact k-nearest-neighbor distance handling;
- CIR-LID, normalized arithmetic, SP-LID, and secondary scores;
- frame/crop/mask geometry utilities;
- gallery and feature artifact persistence;
- threshold calibration, pixel metrics, cluster bootstrap, and gate decisions;
- synthetic manifold nuisance experiment;
- official DINOv3 ViT-B/16 loading through the authorized local repository/checkpoint, ImageNet preprocessing, and block-6 patch-token smoke validation.

The crop stage writes crop and mask artifact paths back into both manifests;
these fields are intentionally preserved when the JSONL/CSV mirrors are
rewritten.  Validation-real and validation-fake frame positions are read
separately from the frozen YAML rather than being silently replaced by the
test positions.

The FF++ stages only read the already-authorized local dataset; they never download or
modify raw videos. `mask_binarization: nonzero` is an explicit protocol revision based
on the local server audit of grayscale/color FF++ masks and must be recorded in the
experiment log before final results are reported.

## Current integration boundaries

The following local stages are implemented and ready to run on the server after
the documented dependencies and external artifacts are present:

- GPU feature extraction for `gallery`, `validation`, and gated `test` subsets;
- exact train-real/c23 gallery construction and provenance checks;
- validation-real-only threshold calibration;
- the synthetic anomaly × nuisance experiment and synthetic Gate B artifact;
- resumable crop/mask caching with atomic array writes and exclusion logging;
- the isolated M0-v1 real-only static manifold audit (`python -m src.run_m0`),
  which never enumerates fake media or opens the official test split.

The following remains intentionally pending because it requires the complete
server-side FF++ feature/result orchestration and paired bootstrap outputs:

- Gate A-α and Gate A-β end-to-end command orchestration;
- validation fake c23→c40 Gate B pilot orchestration;
- combined gate decision, Gate D, final bootstrap, and figure commands.

The intended entry points are the modules listed in `CIR-LID_v5.5_实验说明.md` section 20.
All commands accept `--config` and use the frozen YAML configuration as their single
source of truth.
