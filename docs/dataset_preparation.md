# Rebuilding benchmark_v5

benchmark_v4 is the frozen historical benchmark used for completed YOLO11m-seg and Mask2Former experiments. Do not regenerate it, alter its split, or overwrite its experiment results. The legacy prepare entry point refuses to write benchmark_v4.

benchmark_v5 discovers all descendants of the immutable raw annotation root in `config/dataset_sources.yaml`, including Jakub. This tracked file defines the raw root and existing zmladenie/circle exclusions. Unsupported labels, non-polygon shapes, malformed annotations, mismatched dimensions, ambiguous pairs and invalid polygons exclude the entire image with an explicit reason. Empty LabelMe annotations are valid negative images. The existing parser's at-most-one-pixel boundary normalization is retained only in generated labels; raw JSON is never edited.

Run from the repository root using an environment with requirements.txt installed:

```powershell
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --audit-only
python main.py --config config/config.yaml prepare --dataset benchmark_v5
# Explicit replacement of existing generated v5 output:
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --rebuild
```

Audit reports are written to `results/benchmark_v5_audit/`: `audit.json` (every folder, orphan/invalid JSON lists, duplicate groups and filename grouping evidence), `provenance.json` (every image and its disposition), and `annotation_conflicts.json` (all variants, labels, counts and geometry differences). Audit-only never replaces staging or a benchmark. Unresolved annotation conflicts block generation even with --rebuild and preserve existing output. Resolve conflicts through an explicitly reviewed future source-selection rule; this version intentionally provides no automatic winner or force-conflicts switch.

Duplicate identity is the SHA256 of the original image bytes, independent of source paths. Annotation comparison ignores label case, polygon starting vertex, winding and shape ordering, while preserving geometry, instance grouping and flags. Different encodings of the same pixels are not deduplicated. Conflicting excluded variants also block selection of eligible alternatives. Identical variants select the first lexically sorted relative source path. No v4 annotation preference is reused.

On a safe build, `data_staging/benchmark_v5/` contains byte-for-byte images and JSON plus `provenance.json`. `datasets/benchmark_v5/` contains train/val/test images and YOLO segmentation labels, `dataset.yaml`, COCO `coco/annotations/instances_{train,val,test}.json`, `coco/dataset_fingerprint.txt`, `dataset_manifest.csv`, `dataset_summary.json`, and `validation.json`. Every directory is generated and may be deleted and rebuilt. Rebuild audits before replacing these exact v5 directories; an interrupted build can be rerun with --rebuild.

The full SHA256 serves as the canonical image ID. Sorted IDs are shuffled with Python random.Random(42). Train and val counts are floor(N*0.70) and floor(N*0.15); test receives the remainder. Both model formats share that assignment. COCO category IDs 1/2 map to model classes 0 tree / 1 TreeGroup, matching the existing Mask2Former adapter. Annotation geometry is validated before inclusion and again after YOLO rounding. Validation checks hashes, distinct IDs/content, annotation counts, references, categories, polygons, and exact YOLO/COCO agreement.

The manifest retains original paths, annotation SHA256, source identities and matching v4 IDs/splits found by content hash. No v4 files are written. Content IDs, decisions and split assignment are reproducible for unchanged inputs and rules; generation timestamps and absolute paths depend on the execution environment.

Filenames with numeric tile suffixes are reported as candidate groups, not asserted to be spatially independent. Confirm original imagery/flight identities before adopting grouped splitting. SPATIAL LEAKAGE STATUS: UNVERIFIED. A grouped split is preferable if parent-image relationships are confirmed.

Future model comparisons must train BOTH YOLO and Mask2Former on v5. Use a copy of the training configuration with dataset.output_dir set to datasets/benchmark_v5 and separate experiment directories. The historical default config remains pointed at v4.

Git tracks source code, source definitions, exclusion policy, documentation and the lightweight first audit summary. Copied imagery, staging JSON, generated labels, model weights and large output reports stay ignored. Full provenance reports remain local because of their size and absolute source paths.

The initial audit is recorded in [benchmark_v5_initial_audit.md](benchmark_v5_initial_audit.md). Matej's corrected image+JSON pairs are assessed individually: incomplete sibling folders do not invalidate matched, valid pairs. Orphan-only annotation folders remain explicitly reported. This replaces the earlier whole-source omission of Matej with per-image eligibility, preserving the label/shape exclusions.

Lightweight v5 manifest, summary and validation files are allowed by the root .gitignore; generated imagery and labels remain ignored.
