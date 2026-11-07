"""Build an immutable, local static classroom comparison site. Never runs inference."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import parse_qs, quote, unquote, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = "class-20261108"
LABELS = {"tree", "treeGroup"}
ASSETS = Path(__file__).with_name("classroom_viewer_assets")


class ViewerError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ViewerError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filename(value):
    require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]*\.(?i:jpg|jpeg)", value))
            and ".." not in value, f"Unsafe image filename: {value!r}")
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def geometry(points, width, height):
    require(isinstance(points, list) and len(points) >= 3, "Polygon must have at least three vertices")
    require(all(isinstance(p, list) and len(p) == 2 and all(number(v) for v in p)
                and 0 <= p[0] <= width and 0 <= p[1] <= height for p in points),
            f"Invalid polygon coordinates for {width} × {height}")
    return points


def prediction_index(data, expected, dimensions, stage):
    require(data.get("schema_version") == 1 and data.get("run_id") == RUN and data.get("stage") == stage,
            f"Wrong prediction schema, run ID or stage ({stage})")
    require(data.get("coordinate_system", "").startswith("original-image pixels"),
            "Predictions must use original-image pixel coordinates")
    require(isinstance(data.get("model", {}).get("id"), str), "Missing model checkpoint identifier")
    index = {}
    for record in data["images"]:
        name = filename(record["filename"])
        require(name not in index and name in expected, f"Duplicate or unexpected model image: {name}")
        require(record["image_sha256"] == expected[name], f"Prediction hash mismatch: {name}")
        require((record["width"], record["height"]) == dimensions[name], f"Prediction dimensions mismatch: {name}")
        ids = set()
        for prediction in record["predictions"]:
            require(prediction["class"] in LABELS, f"Unknown class: {name}")
            require(number(prediction["confidence"]) and 0 <= prediction["confidence"] <= 1,
                    f"Invalid confidence: {name}")
            require(prediction["model_checkpoint_id"] == data["model"]["id"], f"Checkpoint mismatch: {name}")
            require(prediction["id"] not in ids, f"Duplicate polygon ID: {name}")
            ids.add(prediction["id"])
            geometry(prediction["polygon"], *dimensions[name])
        index[name] = record
    require(set(index) == set(expected) and data["image_count"] == len(expected),
            f"Missing model prediction records ({stage}): {sorted(set(expected) - set(index))}")
    return index


def validate_inputs(plan, state, baseline, images, finetuned=None):
    require(plan.get("run_id") == state.get("run_id") == RUN, "Plan/state run ID mismatch")
    students = plan["students"]
    require(len(students) == 42, "Expected exactly 42 students")
    aliases, projects, assignments = set(), set(), []
    for student in students:
        alias = student["alias"]
        require(bool(re.fullmatch(r"s\d{3}", alias)) and alias not in aliases, f"Invalid/duplicate alias: {alias}")
        aliases.add(alias)
        require(isinstance(student["name"], str) and student["name"].strip(), f"Missing student name: {alias}")
        names = student["images"]
        require(len(names) == len(set(names)) == 5, f"{alias}: expected exactly five distinct images")
        assignments.extend(filename(n) for n in names)
        record = state["students"].get(alias, {})
        project = record.get("project_id")
        require(type(project) is int and project > 0 and project not in projects, f"Invalid/duplicate project: {alias}")
        require(type(record.get("user_id")) is int, f"Missing user ID: {alias}")
        projects.add(project)
        require(record.get("title") == student["project_title"] and record.get("username") == student["username"],
                f"Plan/state identity mismatch: {alias}")
    require(set(state["students"]) == aliases, "State student aliases differ from plan")
    counts = Counter(assignments)
    require(len(assignments) == 210 and len(counts) == 200, "Expected 210 assignments / 200 distinct images")
    require(Counter(counts.values()) == {1: 190, 2: 10}, "Expected exactly ten images shared by two students")
    require(plan.get("assignments") == 210 and plan.get("distinct_images") == 200
            and plan.get("images_per_student") == 5, "Plan summary counts disagree")
    require(set(plan["image_hashes"]) == set(counts), "Plan image hash inventory differs from assignments")
    dimensions = {}
    root = images.resolve(strict=True)
    source_names = {p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}}
    require(source_names == set(counts), "Source directory must contain exactly the 200 planned JPG images")
    for name in counts:
        path = root / name
        require(not path.is_symlink() and path.resolve(strict=True).parent == root, f"Image escapes source directory: {name}")
        require(digest(path) == plan["image_hashes"][name], f"Source image hash mismatch: {name}")
        with Image.open(path) as image:
            require(image.format == "JPEG" and image.getexif().get(274, 1) == 1,
                    f"Unsupported image format/orientation: {name}")
            dimensions[name] = image.size
    base = prediction_index(baseline, plan["image_hashes"], dimensions, "baseline")
    fine = prediction_index(finetuned, plan["image_hashes"], dimensions, "finetuned") if finetuned else None
    return base, fine, dimensions


def task_filename(value, run_id=RUN):
    require(isinstance(value, str), "Label Studio task has no image URL")
    url = urlsplit(value)
    query = parse_qs(url.query)
    path = unquote(query["d"][0]) if "d" in query and len(query["d"]) == 1 else unquote(url.path)
    parts = path.replace("\\", "/").split("/")
    require(".." not in parts and not any("\x00" in p for p in parts), "Unsafe Label Studio image path")
    if "classroom_runs" in parts:
        i = parts.index("classroom_runs")
        require(parts[i + 1:-1] == [run_id, "images"], "Label Studio image belongs to another classroom run")
    return filename(parts[-1])


def annotation_order(annotation):
    stamp = annotation.get("updated_at") or annotation.get("created_at")
    try:
        time = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() if stamp else 0
    except (ValueError, AttributeError) as exc:
        raise ViewerError("Invalid annotation timestamp") from exc
    annotation_id = annotation.get("id", "")
    return time, annotation_id if type(annotation_id) is int else -1, str(annotation_id)


def import_annotations(directory, plan, state, dimensions):
    """Return per-student submitted task annotations. None means unavailable, [] submitted empty."""
    result = {s["alias"]: {} for s in plan["students"]}
    if directory is None:
        return result
    directory = directory.resolve(strict=True)
    expected = {s["alias"] + "-annotations.json" for s in plan["students"]}
    require(not ({p.name for p in directory.glob("*-annotations.json")} - expected), "Unknown student export file")
    complete = load(directory / "complete.json") if (directory / "complete.json").exists() else None
    if complete:
        require(complete.get("run_id") == RUN, "Archive run ID mismatch")
        for name, checksum in complete["sha256"].items():
            require(name in expected, f"Unexpected archive export: {name}")
            path = directory / name
            require(path.is_file() and not path.is_symlink() and path.resolve().parent == directory
                    and digest(path) == checksum, f"Archive checksum mismatch: {name}")
        require(set(complete["sha256"]) == {p.name for p in directory.glob("*-annotations.json")},
                "Archive checksum inventory differs from exports")
    for student in plan["students"]:
        alias = student["alias"]
        path = directory / f"{alias}-annotations.json"
        if not path.exists():
            continue
        require(not path.is_symlink() and path.resolve().parent == directory, "Export escapes annotation directory")
        tasks = load(path)
        require(isinstance(tasks, list), f"{alias}: expected Label Studio JSON task list")
        seen = set()
        for task in tasks:
            project = task.get("project")
            if project is not None:
                require(project == state["students"][alias]["project_id"], f"{alias}: wrong export project ID")
            name = task_filename(task["data"]["image"])
            require(name in student["images"] and name not in seen, f"{alias}: duplicate/unassigned task {name}")
            seen.add(name)
            annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
            if not annotations:
                continue  # predictions and drafts are not student submissions
            annotation = max(annotations, key=annotation_order)
            width, height = dimensions[name]
            polygons = []
            for region in annotation["result"]:
                require(region.get("type") == "polygonlabels", f"{alias}/{name}: unsupported annotation type")
                require(region.get("from_name") == "label" and region.get("to_name") == "image",
                        f"{alias}/{name}: wrong Label Studio control")
                require(region.get("image_rotation", 0) == 0, "Rotated Label Studio images are unsupported")
                require((region.get("original_width"), region.get("original_height")) == (width, height),
                        f"{alias}/{name}: annotation dimensions differ from source")
                value = region["value"]
                require(value.get("polygonlabels") in (["tree"], ["treeGroup"]), "Unknown student polygon label")
                points = geometry(value["points"], 100, 100)
                pixels = [[x * width / 100, y * height / 100] for x, y in points]
                polygons.append({"class": value["polygonlabels"][0], "polygon": geometry(pixels, width, height)})
            result[alias][name] = polygons
    return result


def esc(value):
    return html.escape(str(value), quote=True)


def shell(title, content):
    return f'''<!doctype html>
<html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Tree AI</title><link rel="stylesheet" href="assets/viewer.css">
<script src="assets/viewer.js" defer></script></head><body>
<header><a class="brand" href="index.html"><span class="brand-icon">♧</span> TREE AI <small>UNIVERZITNÍ CVIČENÍ</small></a><span class="run">{RUN}</span></header>
<main>{content}</main><footer>Tree AI · Porovnání anotací korun stromů · Pouze pro čtení</footer>
<dialog id="enlarged"><button class="close" autofocus>Zavřít ×</button><div class="dialog-content"></div></dialog>
</body></html>'''


def project_link(state, alias):
    base = state.get("base_url", "")
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        return ""
    project = state["students"][alias]["project_id"]
    return f'<a class="studio" href="{esc(base.rstrip("/") + f"/projects/{project}/data")}" target="_blank" rel="noopener noreferrer">Label Studio ↗</a>'


def panel(name, record, kind, predictions):
    titles = {"student": "Anotace studenta", "baseline": "Původní TreeDetector", "finetuned": "Doučený TreeDetector"}
    width, height = record["width"], record["height"]
    polygons = []
    details = []
    for prediction in predictions or []:
        label = prediction["class"]
        confidence = f' · {prediction["confidence"]:.1%}' if "confidence" in prediction else ""
        title = esc(label + confidence)
        points = " ".join(f"{x:.6f},{y:.6f}" for x, y in prediction["polygon"])
        group = ' class="tree-group"' if label == "treeGroup" else ""
        polygons.append(f'<polygon{group} points="{points}" vector-effect="non-scaling-stroke"><title>{title}</title></polygon>')
        details.append(f'<li>{title}</li>')
    if predictions is None:
        status = "Zatím neanotováno" if kind == "student" else "Model zatím nebyl doučen"
    elif not predictions:
        status = "Anotováno bez polygonů" if kind == "student" else "Bez detekcí"
    else:
        status = f"Počet polygonů: {len(predictions)}"
    return f'''<section class="panel {kind}" aria-label="{titles[kind]}"><h3><span class="dot"></span>{titles[kind]}</h3>
<figure><svg role="img" aria-label="{esc(name)} – {titles[kind]}" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<image href="images/{quote(name)}" width="{width}" height="{height}"/><g class="polygons">{"".join(polygons)}</g></svg>
<button class="enlarge" type="button" aria-label="Zvětšit: {esc(name)} – {titles[kind]}">Zvětšit ↗</button></figure>
<p class="status">{status}</p><details><summary>Třídy a jistota ({len(details)})</summary><ul>{"".join(details) or '<li>Žádné polygony k zobrazení</li>'}</ul></details></section>'''


def student_page(student, state, base, fine, annotations, counts, available_exports=()):
    alias = student["alias"]
    submitted = annotations[alias]
    completed = len(submitted)
    progress = f"{completed} / 5 anotováno" if completed or alias in available_exports else "Anotace zatím nejsou k dispozici"
    images = []
    for i, name in enumerate(student["images"], 1):
        shared = '<span class="badge">Sdílený snímek</span>' if counts[name] == 2 else ""
        record = base[name]
        panels = panel(name, record, "student", submitted.get(name))
        panels += panel(name, record, "baseline", record["predictions"])
        panels += panel(name, record, "finetuned", fine[name]["predictions"] if fine else None)
        images.append(f'<article class="comparison" data-filename="{esc(name)}" data-image-sha256="{record["image_sha256"]}"><div class="image-heading"><h2><span>{i:02}</span> {esc(name)}</h2>{shared}<small>{record["width"]} × {record["height"]} px</small></div><div class="panels">{panels}</div></article>')
    return shell(student["name"], f'''<a class="back" href="index.html">← Zpět na seznam studentů</a>
<div class="page-heading"><div><p class="eyebrow">STUDENT / {alias.upper()}</p><h1>{esc(student["name"])}</h1><p class="subtitle">5 přidělených snímků · {progress}</p></div>{project_link(state, alias)}</div>
<div class="toolbar"><label><input id="show-polygons" type="checkbox" checked> Zobrazit polygony</label><span>Plná čára: tree · Přerušovaná čára: treeGroup</span></div>
<p class="notice">Chybějící anotace není chyba modelu. Průběh vychází pouze z dostupných odevzdaných anotací.</p>{"".join(images)}''')


def index_page(students, state, annotations, preview, available_exports=()):
    cards = []
    for student in students:
        alias = student["alias"]
        completed = len(annotations[alias])
        progress = f"{completed} / 5 anotováno" if completed or alias in available_exports else "Průběh zatím není k dispozici"
        cards.append(f'''<li class="student-card" data-name="{esc(student['name'])}"><div class="card-top"><span class="alias">{alias.upper()}</span><span class="assignment">5 snímků</span></div>
<h2><a href="{alias}.html">{esc(student['name'])}</a></h2><p class="progress">{progress}</p><div class="card-bottom"><a class="compare-link" href="{alias}.html">Otevřít porovnání →</a>{project_link(state, alias)}</div></li>''')
    return shell("Přehled studentů", f'''<p class="eyebrow">KORUNY STROMŮ / ANOTAČNÍ CVIČENÍ</p><div class="page-heading"><div><h1>Jeden snímek.<br>Tři pohledy na koruny.</h1><p class="subtitle">Porovnejte studentské anotace s původním a doučeným modelem TreeDetector.</p></div><span class="read-only">Pouze pro čtení</span></div>
{'<p class="notice">Náhled jednoho studenta — úplný plán byl ověřen.</p>' if preview else ''}
<div class="stats"><div><strong>{len(students)}</strong><span>studentů</span></div><div><strong>{len(students)*5}</strong><span>přidělení snímků</span></div><div><strong>200</strong><span>unikátních snímků v plánu</span></div><div><strong>10</strong><span>sdílených snímků v plánu</span></div></div>
<div class="list-heading"><h2>Studenti</h2><label class="search">Vyhledat studenta <input type="search" id="student-search" placeholder="Jméno studenta…"></label></div>
<p class="notice">Zelená: student · Modrá: původní model · Fialová: doučený model. Chybějící anotace není chyba modelu.</p>
<ul class="students">{"".join(cards)}</ul><p id="empty-search" hidden>Žádný student neodpovídá hledání.</p>''')


def protect_output(output, args):
    resolved = output.resolve()
    require(not output.exists() and not output.is_symlink(), f"Output already exists; choose a new directory: {output}")
    inputs = [args.plan, args.state, args.baseline, args.images]
    inputs += [p for p in (args.annotations, args.finetuned) if p]
    protected = [ROOT / p for p in ("runs", "datasets", "images", "weights", ".git")]
    for path in inputs + protected:
        source = path.resolve()
        require(resolved != source and source not in resolved.parents and resolved not in source.parents,
                f"Output overlaps protected input: {path}")


def build(args):
    protect_output(args.output, args)
    plan, state, baseline = load(args.plan), load(args.state), load(args.baseline)
    fine_data = load(args.finetuned) if args.finetuned else None
    base, fine, dimensions = validate_inputs(plan, state, baseline, args.images, fine_data)
    annotations = import_annotations(args.annotations, plan, state, dimensions)
    available_exports = {s["alias"] for s in plan["students"]
                         if args.annotations and (args.annotations / f'{s["alias"]}-annotations.json').is_file()}
    students = plan["students"]
    if args.student:
        students = [s for s in students if s["alias"] == args.student]
        require(len(students) == 1, f"Unknown student: {args.student}")
    counts = Counter(n for s in plan["students"] for n in s["images"])
    used = {n for s in students for n in s["images"]}
    # Render and validate all data before creating the new output directory.
    pages = {s["alias"] + ".html": student_page(s, state, base, fine, annotations, counts, available_exports) for s in students}
    pages["index.html"] = index_page(students, state, annotations, bool(args.student), available_exports)
    input_files = [args.plan, args.state, args.baseline] + ([args.finetuned] if args.finetuned else [])
    if args.annotations:
        input_files += sorted(args.annotations.glob("*.json"))
    snapshots = {str(p.resolve()): digest(p) for p in input_files}
    report = {"run_id": RUN, "student_pages": len(students), "assignments": len(students)*5,
              "source_images": len(used), "full_plan_students": 42, "full_plan_assignments": 210,
              "full_plan_distinct_images": 200, "shared_images": sorted(n for n, c in counts.items() if c == 2),
              "baseline_model": baseline["model"]["id"],
              "finetuned_model": fine_data["model"]["id"] if fine_data else None,
              "submitted_tasks": {s["alias"]: len(annotations[s["alias"]]) for s in students},
              "input_sha256": snapshots}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    (args.output / "assets").mkdir()
    for name in sorted(used):
        destination = args.output / "images" / name
        shutil.copyfile(args.images / name, destination)
        require(digest(destination) == plan["image_hashes"][name], f"Copied image hash mismatch: {name}")
    for name in ("viewer.css", "viewer.js"):
        shutil.copyfile(ASSETS / name, args.output / "assets" / name)
    for name, content in pages.items():
        (args.output / name).write_text(content, encoding="utf-8")
    require(all(digest(Path(p)) == checksum for p, checksum in snapshots.items()), "Source JSON changed during generation")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validated full plan: 42 students / 210 assignments / 200 images / 10 shared images")
    print(f"Generated {len(students)} student pages and index: {args.output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run = ROOT / "outputs" / RUN
    parser.add_argument("--plan", type=Path, default=run / "plan.json")
    parser.add_argument("--state", type=Path, default=run / "state.json")
    parser.add_argument("--baseline", type=Path, default=run / "baseline-200.json")
    parser.add_argument("--images", type=Path, default=ROOT / "images/randomSelect")
    parser.add_argument("--output", type=Path, default=run / "viewer-v1")
    parser.add_argument("--annotations", type=Path, help="Archive directory containing sNNN-annotations.json")
    parser.add_argument("--finetuned", type=Path, help="Optional full prediction JSON with stage=finetuned")
    parser.add_argument("--student", help="Generate one student's preview after validating the entire plan")
    args = parser.parse_args()
    try:
        build(args)
    except (ViewerError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
