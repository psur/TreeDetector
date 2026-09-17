# benchmark_v5 final human review and build

Open `results/benchmark_v5_review/index.html`. Every item shows its stable
`review_id`, source/counts, overlays, and a reference to its initial Excel row.
Geometry pages show severity, one-based polygon index, highlighted contour,
and an enlarged crop. Match by review_id after sorting the spreadsheet.

## Excel workflow

Open `results/benchmark_v5_review/human_review_required.csv` in Excel. It has a
UTF-8 BOM and semicolon separators for the local Excel locale. If necessary,
use Data → From Text/CSV, UTF-8, delimiter Semicolon. Save as CSV UTF-8.
The validator accepts both semicolon and comma CSVs. Never save the workbook's
binary XLSX bytes under a CSV extension.

Only enter human choices; `decision_status` is computed by validation:

- Conflicts: `KEEP_A`, `KEEP_B`, `EXCLUDE`.
- Self-intersections: `KEEP_AS_IS`, `AUTO_FIX_APPROVED`, `MANUAL_FIX`, `EXCLUDE`.

For MANUAL_FIX, `selected_annotation` must name a completed annotation JSON
under `data_staging/benchmark_v5/`. Its `.provenance.json` sidecar must contain
`original_annotation`, `original_annotation_sha256`, `fixed_annotation_sha256`,
and `content_sha256`. The entire copy must validate. It is never stored in raw.
For other geometry decisions, selected_annotation stays empty. KEEP_A/B may
leave it empty; if supplied, it must match the specified alternative.

`AUTO_FIX_APPROVED` is supported for the 9 MINOR single proper crossings listed
as `auto_fix_supported=YES`. The deterministic operation reverses the vertex
subsequence between the crossing edges (one 2-opt operation); it keeps every
vertex coordinate and must produce a valid simple polygon before and after
export rounding. Border touches, overlaps, multiple crossings, ERRONEOUS and
AMBIGUOUS cases are unsupported. Unsupported approval is INVALID. Validation
only computes a candidate in memory. Approved repairs are written to staging
only by a subsequent build. No such decisions or repairs have been made now.

`KEEP_AS_IS` records an explicit polygon-specific self-intersection exception;
it does not label the polygon valid. Zero-area/out-of-bounds/unsupported shapes
are still rejected. COCO uses the same serialized polygon coordinates and
absolute shoelace area as YOLO; an accepted self-intersection can have different
rasterized area. The summary records all such exceptions for downstream users.

## Validate and finalize

Validate the canonical file:

```powershell
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --validate-decisions
```

While editing the review spreadsheet, validate that file explicitly:

```powershell
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --validate-decisions --decisions results/benchmark_v5_review/human_review_required.csv
```

The command reports TOTAL REVIEW ITEMS, COMPLETE, PENDING, INVALID, conflict/
geometry completion, and SAFE TO BUILD. If unsafe, it prints only unresolved
IDs after the summary. Detailed row errors are saved to
`results/benchmark_v5_review/decision_validation.json`; the ID-only list is
`unresolved_review_ids.txt`. It updates computed status cells, preserving human
values. Missing IDs and blank decisions are PENDING. Unknown/duplicate IDs,
identity changes, incompatible choices, unsupported repairs, bad manual-copy
provenance, and changed sources are INVALID. Every required ID must be present
and filled, including other polygon rows for an excluded content group.

When the review spreadsheet is complete, explicitly finalize it:

```powershell
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --finalize-decisions --decisions results/benchmark_v5_review/human_review_required.csv
```

Finalization refuses incomplete/invalid input. It saves the previous and new
canonical CSVs by SHA-256 under `config/benchmark_v5_decision_history/`, then
imports the actual human values into `config/benchmark_v5_decisions.csv`.
It never chooses a decision. Alternatively, edit the canonical file directly.

The canonical file contains review_id, issue_type, content_sha256,
human_decision, selected_annotation, notes, and computed decision_status.
It and the canonical review catalog/source snapshot are Git-trackable. The
build never reads human_review_required.csv or depends on results-folder CSVs.
After initialization, the older package generator refuses to overwrite final
review spreadsheets. Preserve the canonical file in Git after reviewing it.

## Deterministic build

Only after validation succeeds:

```powershell
python main.py --config config/config.yaml prepare --dataset benchmark_v5 --rebuild
```

A pending/invalid file blocks before dataset or generation staging writes.
The command re-reads all source images and JSONs, compares hashes and file sets
against `config/benchmark_v5_source_snapshot.json` (893 files, including orphan
JSONs), and checks alternatives against `benchmark_v5_review_catalog.json`.
Worldfiles, XML sidecars, and raster overview caches are not build inputs.

Precedence is deterministic:

1. An explicit EXCLUDE overrides source selection and repairs for that content.
2. Human KEEP_A/B selects the conflict source. No annotations are merged.
3. Only the selected source is used. Its safe consecutive duplicates are removed
   in staging; rounding-only duplicates are removed only in generated coordinates.
4. Selected self-intersections require the explicit geometry decisions above.
5. Deduplicate content, split with seed 42, export YOLO and COCO, validate, then
   publish the completed candidate. Raw data and benchmark_v4 remain untouched.

The split keeps inferred parent-frame filename groups together globally across
annotators and folders. Targets are 70/15/15, with actual counts determined by
group sizes. Validation checks unique content hashes, decoded pixel duplicates,
parent-group leakage, labels, counts, and exact COCO/YOLO geometry agreement.
Flight-level/spatial independence remains unverified without authoritative
metadata; the output reports that limitation rather than claiming it verified.

Approved generation artifacts and provenance remain under
`data_staging/benchmark_v5/approved_generations/`. Existing safe-repair and manual
staging files are preserved. Validation completes before replacing an existing
benchmark_v5; the previous dataset is retained in a sibling backup directory.
Every generated annotation binds original path/hash, staged hash, repair events,
and decision checksum. `dataset_summary.json` records the canonical decision
checksum, catalog/snapshot checksums, generator/validator checksums, seed, split
method, and any explicitly retained self-intersections. The exact decision file
is also copied into the generated dataset.

The canonical file currently has 67 blank decisions: 39 conflicts and 28
self-intersection polygons. Seventeen safe duplicate repairs remain staged and
require no additional human decisions. No real benchmark_v5 has been generated.
