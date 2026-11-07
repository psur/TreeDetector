# Read-only classroom comparison viewer

The Czech static viewer for `class-20261108` is generated at
`outputs/class-20261108/viewer-v1/index.html`.

## Open locally

From `D:\TreeDetector`:

```powershell
.venv-prep/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1 --directory outputs/class-20261108/viewer-v1
```

Open **http://127.0.0.1:8000/**. Stop the server with Ctrl+C.
The generated site needs no Python backend, framework, CDN, or API access;
any static HTTP server can serve the entire directory.

The index lists all 42 names from the plan, their five assignments, available
submission progress, comparison links, and project links derived from state.json.
Search ignores Czech diacritics. Each student page has five image rows with
three panels showing the same image. Toggle polygons with the checkbox, enlarge
any panel with **Zvětšit**, and return via **Zpět na seznam studentů**. The enlarged
view closes with the close button, Escape, or a click outside the dialog.

Student polygons are green, original-model polygons blue, and fine-tuned polygons
purple. `treeGroup` has a dashed outline. Tooltips and the expandable **Třídy a
jistota** list show classes and model confidences. Confidence is the model's
score, not an annotation agreement or accuracy measurement.

## Generated files

```text
outputs/class-20261108/viewer-v1/
  index.html
  s001.html ... s042.html       # exactly 42 student pages
  images/                     # 200 independent JPEG copies, one per source filename
  assets/viewer.css
  assets/viewer.js
  validation.json             # counts, shared images, models, source JSON hashes
```

The first-student preview is separately retained at
`outputs/class-20261108/viewer-preview-s001/`. It has one student page and five
image assets. Original files remain in place and are never used as write targets.
No images are embedded as base64. No state.json, emails, tokens, or raw Label Studio
exports are copied to the website. Model polygons are rendered into SVG; shared
images use the same validated baseline record during generation and one copied
asset. Student overlays are keyed separately by student alias and image filename.

## Generate a new version

```powershell
# Validate the entire input inventory, then generate only the first student:
.venv-prep/Scripts/python.exe scripts/build_classroom_viewer.py --student s001 --output outputs/class-20261108/viewer-preview-s001

# Full generation (already executed for viewer-v1):
.venv-prep/Scripts/python.exe scripts/build_classroom_viewer.py
```

Existing output directories are refused. Use a new `--output` for every revision;
the two commands above cannot overwrite the already generated versions.
Inputs default to:

- `outputs/class-20261108/plan.json`
- `outputs/class-20261108/state.json`
- `outputs/class-20261108/baseline-200.json`
- `images/randomSelect/`

Override these via `--plan`, `--state`, `--baseline`, and `--images` if needed.
The generator does not import or invoke training/inference code.

### After student annotations arrive

Point `--annotations` at the **single archive snapshot directory** containing
`s001-annotations.json`, `s002-annotations.json`, etc. Do not point it at an
archive root containing multiple dated snapshots.

```powershell
.venv-prep/Scripts/python.exe scripts/build_classroom_viewer.py --annotations D:/path/to/archive/class-20261108/TIMESTAMP --output outputs/class-20261108/viewer-v2
```

The importer accepts the JSON task lists produced by the existing classroom.py
archive command. If `complete.json` exists, its run ID, export inventory, and
SHA-256 checksums must match. Partial export directories without `complete.json`
are also supported: missing student files remain unavailable.

Each task is associated with its student by export filename and with its image
by `data.image`. Label Studio local-file `?d=classroom_runs/.../images/...` URLs
are decoded and their classroom run checked; plain image paths/URLs are also
supported. When a task includes a project ID, it must match state.json. Exports
must contain no duplicate tasks or images outside that student's five assignments.

Only submitted, non-cancelled `annotations` are imported. Drafts and model
`predictions` in Label Studio exports are ignored. If several submitted annotations
exist for a task, the latest `updated_at` / `created_at` wins, with annotation ID
as a deterministic tie-break. Supported regions are unrotated `polygonlabels`
from control `label` to image `image`, with `tree` or `treeGroup`, original
dimensions matching the image, and points in Label Studio's 0–100 percentage units.
Unsupported region types or rotations fail explicitly.

Percentage coordinates are converted once to original-image pixels:
`x_px = x_percent * width / 100`, `y_px = y_percent * height / 100`.
The importer never writes into the archive. Standard Label Studio exports do
not carry an image-content hash, so annotation identity depends on the trusted
archive/student mapping and task URL; source pixels and model records are
independently verified against the plan's hashes.

No submitted annotation displays **Zatím neanotováno**. A submitted annotation
with an empty result displays **Anotováno bez polygonů** and counts as submitted.
An available export with zero submissions shows `0 / 5 anotováno`; missing export
files show progress as unavailable. Missing labels are never interpreted as model
false positives. The viewer calculates no false-positive or agreement scores.

### After fine-tuned predictions arrive

Supply a separate, previously generated prediction JSON using the baseline schema,
the same run ID, all 200 filename/hash records, original-image pixel coordinates,
and `stage: "finetuned"`:

```powershell
.venv-prep/Scripts/python.exe scripts/build_classroom_viewer.py --annotations D:/path/to/archive/class-20261108/TIMESTAMP --finetuned D:/path/to/finetuned-200.json --output outputs/class-20261108/viewer-v3
```

The baseline is reused unchanged. Without `--finetuned`, the original image is
still shown in the third panel with **Model zatím nebyl doučen**.

## Integrity and validation

Before creating an output directory, generation verifies:

- Exactly 42 students, five different assignments each, 210 assignments total,
  200 distinct source filenames, and exactly ten images assigned to two students.
- Consistent run IDs, student aliases, usernames/titles and unique project IDs
  across plan and state. Student display names always come from the plan.
- All planned source JPGs are present, with matching SHA-256 hashes and supported
  orientation; no extra JPGs are accepted in the source directory.
- Baseline and optional fine-tuned records match by both filename and SHA-256,
  with no missing or duplicate records, matching original dimensions, valid
  classes, finite in-bounds polygons, confidence values, and checkpoint IDs.
- Safe filenames, aliases, and paths; output cannot overlap inputs or protected
  model/dataset/image directories. Existing output directories are refused.

SVGs use `viewBox="0 0 width height"` and an image of those exact original
dimensions. Model coordinates are not normalized or scaled to the 640-pixel
inference input. Responsive CSS scales the SVG and image together, and polygon
stroke widths remain readable with `vector-effect="non-scaling-stroke"`.

The real first-student page was generated and independently checked before the
full site. A subsequent full audit confirmed **42 student pages, 210 assignments,
200 matching image-asset hashes, 630 aligned SVG panels, and 4,009 baseline polygon
overlays** (shared assignments intentionally display the shared prediction again).
Every rendered baseline vertex was compared to the source JSON within 0.000001
pixels. Names and five-image ordering matched the plan on all student pages.
All original image hashes and recorded input JSON hashes remained unchanged.

Automated tests:

```powershell
.venv-prep/Scripts/python.exe -m pytest tests/test_classroom_viewer.py tests/test_classroom_predictions.py -q
node --test tests/classroom_viewer_controls.test.cjs
node --check scripts/classroom_viewer_assets/viewer.js
```

Results: **39 Python tests and three JavaScript tests passed**. These include
non-square geometry, shared images with different student annotations, missing
annotations, empty submissions, cancellation, latest-submission selection, archive
checksums, future fine-tuned import, full regeneration, invalid hashes/geometry,
HTML escaping, unsafe paths, overwrite rejection, and unchanged source files.
JavaScript controls were tested with a DOM test double.

Static HTTP delivery was checked successfully. Headless Edge crashed in this
environment with an unusable GPU process before rendering, so visual browser QA
remains unverified. No models were trained, fine-tuned, or run for inference.
