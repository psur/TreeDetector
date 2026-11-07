# Mask2Former v5 preflight fix

Root cause: validate_dataset defaulted to v4 image counts 115/25/25 even when
reading the correct v5 manifest and COCO files. Mask2FormerModel.prepare calls
this function without explicit expected_counts, so 183 train images triggered
the generic count-or-identity error before any actual identity failure occurred.

The default now derives counts from the canonical manifest. Explicit caller
counts are still enforced against that manifest. Canonical identity equality,
duplicate checks, image-file equality, fingerprints, dimensions, instance counts,
mask decoding, and leakage checks remain enabled. Error messages now include
manifest/COCO counts, missing/extra IDs, and duplicate canonical-ID counts.

All four representations agree: train 183, val 39, test 40. There are no
manifest-only/COCO-only IDs, filename mismatches, duplicate canonical/numeric
image IDs, hash mismatches, or invalid polygons. All 3802 masks passed the actual
Mask2Former dataset validator in the tree_mask2former environment.

No COCO regeneration or cache invalidation was needed. The dataset adapter
reads the manifest and COCO JSON directly; it has no persistent dataset index.
The three labels.cache files are YOLO caches and were left unchanged. Hugging
Face's cache contains model assets, not this split membership.

Seven synthetic regression tests passed. Hash checks confirm all 1466 protected
v4/v5/staging/decision files and all 893 raw image/JSON files are unchanged.
No model was loaded, and no training or resplitting ran.

Reports: results/mask2former_v5_preflight/dataset_validation.json and
results/mask2former_v5_preflight/integrity_validation.json.
