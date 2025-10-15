# Side-by-side summary: merged improvements

## Baseline Profiling & Budgeting

- Added datasets (VQA v2, COCO captions, navigation instructions), profiling scripts (`profiling.py`, `power_profiler.py`), and an explicit VRAM/power budget table.

## MVP Resource Budget

- Explicit VRAM component targets: Weights 3 GB, Activations 2 GB, KV Cache 1 GB, Workspace 0.5 GB, Container overhead 1.5 GB; Total ≤7 GB; Power headroom 5 W.

## Operator Compatibility & Runtime API

- Operator Compatibility Checklist for TensorRT unsupported ops and fallbacks.
- ROS 2 Inference API Contract (topic, QoS, message shapes, rate limits, test harness).

## Deliverables & Security

- Added Security & Reliability deliverable items (image signing, telemetry/log retention, vulnerability scans).

## Formatting & Linting

- Sanitized tables; removed inline HTML; converted multi-line table cells into lint-friendly forms or per-role subsections where appropriate.

Notes:

- `openvla_prd2.md` has been archived to `docs/drafts/openvla_prd2.md` to preserve the original draft.
- No deletions were applied to the canonical `openvla_prd.md`; changes were additive and merged conservatively.
