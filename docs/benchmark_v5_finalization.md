# benchmark_v5 finalized with strict geometry exclusion

All 28 self-intersection review cases (27 content groups) are EXCLUDE. The 39
reviewed conflict decisions, including selected paths and notes, were retained.
The previous decision file is preserved by checksum in config decision history.
The final decision checksum is `0ad51183d8d6120b1acdca0aa3d95d0d2ef5f92afc216d7825d192e4f3216a80`.

The configuration now enforces `benchmark_v5_strict_geometry: true`.
Remaining geometry review cases cannot use KEEP_AS_IS, AUTO_FIX_APPROVED, or
MANUAL_FIX under this policy. Only the approved exact duplicate-vertex cleanup
is used; no self-intersection repair or exception was used for this build.

All 17 prior safe repairs remain in staging, byte-for-byte unchanged. Thirteen
repair events apply to annotations selected for the final dataset; the other
four belong to excluded or unselected alternatives.

The dataset contains 262 unique images: train 183, val 39, test 40.
There are 3170 Tree and 632 TreeGroup instances. All 3802 polygons pass strict
geometry validation, with exact YOLO/COCO agreement. Canonical and numeric COCO
image IDs are unique across splits. Content hashes and decoded pixel hashes
are unique. No excluded content appears in the manifest.

All 893 raw source image/JSON files, all 339 benchmark_v4 files, and all 65
original safe-repair staging files were checked against pre-build hashes and
are unchanged. Parent-image filename groups do not cross splits. Flight-level
spatial independence remains unverified without authoritative flight metadata.

Artifacts:

- `datasets/benchmark_v5/dataset_manifest.csv`
- `datasets/benchmark_v5/dataset_summary.json`
- `datasets/benchmark_v5/strict_validation.json`
- `results/benchmark_v5_finalization/before.json`
- `results/benchmark_v5_finalization/validation.json`

The before snapshot retains the original conflict rows and pre-build hashes.
The dataset includes an exact copy of the final canonical decision file.
