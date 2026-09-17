# Local benchmark_v5 review application

Start from D:\TreeDetector:

```powershell
.venv-prep\Scripts\python.exe scripts\review_benchmark_v5.py
```

Open http://127.0.0.1:8766. The server binds only to 127.0.0.1. Stop it with
Ctrl+C in its terminal; restart with the same command to resume. If that port
is already occupied, stop the previous instance or use `--port 8767`.
No Node.js, Flask, FastAPI, Excel, or additional dependencies are needed.
The app uses the preparation environment's existing decision validator.

The app loads the 67 stable IDs and existing review overlays, then reads and
validates the canonical CSV. It starts at the first unresolved item. Existing
choices appear explicitly; Previous, Next, the case selector, First unresolved,
and Next unresolved let you revisit them. Click any decision button to replace
that case's existing choice. Other rows are preserved.

Conflict shortcuts: A = KEEP_A, B = KEEP_B, X = EXCLUDE.
Geometry shortcuts: K = KEEP_AS_IS, F = AUTO_FIX_APPROVED, M = MANUAL_FIX,
X = EXCLUDE. Shortcuts do not fire while entering notes/paths, using a select,
viewing an enlarged image, or saving. Click an image for a larger view.

Notes and a manual-copy path are saved with the next decision click. Unsaved
text edits prompt before navigation or closing. KEEP_A/B always store the exact
A/B path from the review catalog. EXCLUDE clears selected_annotation. Geometry
choices retain the exact decision text. MANUAL_FIX can be recorded before a
fixed copy exists, but validation will show INVALID until a valid staged copy
and provenance are supplied. Unsupported AUTO_FIX_APPROVED is also INVALID.
The save confirmation shows these errors, then moves to the next unresolved
item. Invalid cases remain unresolved and are available for later correction.

Each click writes `config/benchmark_v5_decisions.csv` directly with a flushed,
fsynced temporary file and atomic replacement. The previous exact CSV bytes are
retained by hash under `config/benchmark_v5_decision_history/`. A writer lock and
revision check reject stale saves from another browser window or an external
editor. If warned of a stale version, use Reload from disk before deciding again.
Avoid editing the CSV in Excel while using the app. A crash during a save may
leave a `.csv.lock` file; remove that lock only after all app processes are stopped.

The CSV human fields, not browser storage, are authoritative. The app displays
fresh validator results after saving; the CSV's computed decision_status field
can remain PENDING until the standalone validator refreshes metadata. Startup
and read-only browsing do not rewrite the CSV. Validation errors after a durable
save are reported as saved-but-needing-attention, not as a lost decision.

When all items are complete, the app displays REVIEW COMPLETE and this command:

```powershell
.venv-prep\Scripts\python.exe main.py --config config/config.yaml prepare --dataset benchmark_v5 --validate-decisions
```

The application has no build, geometry-repair, annotation-edit, or deletion
endpoint. It serves only the whitelisted local review JPGs and its own UI.
Requests that could change decisions require a per-server token and matching
loopback host/origin. Raw Dropbox annotations and benchmark_v4 are never written.
No automatic self-intersection fixes or benchmark generation run from this UI.
