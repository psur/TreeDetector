import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from scripts.build_classroom_viewer import (
    ViewerError, build, digest, import_annotations, panel, prediction_index,
    protect_output, task_filename, validate_inputs,
)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def classroom(tmp_path_factory):
    root = tmp_path_factory.mktemp("classroom")
    images = root / "images"
    images.mkdir()
    sample = root / "sample.jpg"
    Image.new("RGB", (1200, 300), "#76876a").save(sample)
    raw = sample.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    names = [f"image{i:03}.JPG" for i in range(200)]
    for name in names:
        (images / name).write_bytes(raw)
    assignments = names + names[:10]
    students = [{"alias": f"s{i+1:03}", "name": f"Student {i+1} Žluťoučký",
                 "images": assignments[5*i:5*i+5], "project_title": f"Project {i+1}",
                 "username": f"student{i+1}"} for i in range(42)]
    plan = {"run_id": "class-20261108", "students": students, "image_hashes": dict.fromkeys(names, checksum),
            "assignments": 210, "distinct_images": 200, "images_per_student": 5}
    state = {"run_id": plan["run_id"], "base_url": "http://localhost:8082", "students": {
        s["alias"]: {"project_id": i+3, "user_id": i+4, "title": s["project_title"], "username": s["username"]}
        for i, s in enumerate(students)}}
    baseline = {"schema_version": 1, "run_id": plan["run_id"], "stage": "baseline", "image_count": 200,
                "coordinate_system": "original-image pixels", "model": {"id": "sha256:test"}, "images": [
                    {"filename": name, "width": 1200, "height": 300, "image_sha256": checksum,
                     "predictions": [{"id": f"{name}:0", "class": "treeGroup", "confidence": 0.9,
                                      "model_checkpoint_id": "sha256:test", "polygon": [[800, 100], [1100, 100], [1100, 280]]}]}
                    for name in names]}
    for name, data in (("plan", plan), ("state", state), ("baseline", baseline)):
        write(root / f"{name}.json", data)
    return root, images, plan, state, baseline


def arguments(classroom, output, **kwargs):
    root, images, *_ = classroom
    return argparse.Namespace(plan=root / "plan.json", state=root / "state.json", baseline=root / "baseline.json",
                              images=images, output=output, annotations=kwargs.get("annotations"),
                              finetuned=kwargs.get("finetuned"), student=kwargs.get("student"))


def test_generate_one_then_all_and_preserve_inputs(classroom, tmp_path):
    root, images, plan, state, baseline = classroom
    original = {p: digest(p) for p in [*root.glob("*.json"), *images.iterdir()]}
    args = arguments(classroom, tmp_path / "preview", student="s001")
    assert build(args)["student_pages"] == 1
    page = (args.output / "s001.html").read_text(encoding="utf8")
    assert '<h1>Student 1 Žluťoučký</h1>' in page
    assert page.count('class="comparison"') == 5
    assert page.count("Zatím neanotováno") == page.count("Model zatím nebyl doučen") == 5
    svgs = [ET.fromstring(s) for s in re.findall(r"<svg\b.*?</svg>", page, flags=re.S)]
    assert len(svgs) == 15
    for i in range(5):
        triple = svgs[3*i:3*i+3]
        assert {svg.attrib["viewBox"] for svg in triple} == {"0 0 1200 300"}
        assert {svg.find("image").attrib["href"] for svg in triple} == {"images/" + plan["students"][0]["images"][i]}
        polygon = triple[1].find("g/polygon")
        assert polygon.attrib["class"] == "tree-group"
        assert polygon.attrib["vector-effect"] == "non-scaling-stroke"
        assert polygon.attrib["points"].startswith("800.000000,100.000000")
        assert "90.0%" in polygon.find("title").text
    args.output = tmp_path / "full"
    args.student = None
    report = build(args)
    assert report["student_pages"] == 42 and report["assignments"] == 210 and report["source_images"] == 200
    assert len(list(args.output.glob("s*.html"))) == 42
    assert len(list((args.output / "images").iterdir())) == 200
    assert (args.output / "index.html").read_text(encoding="utf8").count('class="student-card"') == 42
    assert original == {p: digest(p) for p in original}
    with pytest.raises(ViewerError, match="already exists"):
        build(args)


@pytest.mark.parametrize("problem", ["student-count", "assignment-count", "duplicate-assignment", "alias", "path", "hash", "missing-prediction", "prediction-hash", "dimensions", "geometry", "project", "state-run", "prediction-run", "coordinates"])
def test_reject_data_mismatches(classroom, problem):
    _, images, original_plan, original_state, original_baseline = classroom
    plan, state, baseline = map(deepcopy, (original_plan, original_state, original_baseline))
    if problem == "student-count": plan["students"].pop()
    elif problem == "assignment-count": plan["students"][0]["images"].pop()
    elif problem == "duplicate-assignment": plan["students"][0]["images"][1] = plan["students"][0]["images"][0]
    elif problem == "alias": plan["students"][0]["alias"] = "../index"
    elif problem == "path": plan["students"][0]["images"][0] = "../image.JPG"
    elif problem == "hash": plan["image_hashes"]["image000.JPG"] = "bad"
    elif problem == "missing-prediction": baseline["images"].pop()
    elif problem == "prediction-hash": baseline["images"][0]["image_sha256"] = "bad"
    elif problem == "dimensions": baseline["images"][0]["width"] = 640
    elif problem == "geometry": baseline["images"][0]["predictions"][0]["polygon"][0][0] = 1201
    elif problem == "project": state["students"]["s001"]["project_id"] = state["students"]["s002"]["project_id"]
    elif problem == "state-run": state["run_id"] = "wrong"
    elif problem == "prediction-run": baseline["run_id"] = "wrong"
    elif problem == "coordinates": baseline["coordinate_system"] = "normalized"
    with pytest.raises(ViewerError):
        validate_inputs(plan, state, baseline, images)


def task(name="image000.JPG", polygons=True):
    region = {"type": "polygonlabels", "from_name": "label", "to_name": "image",
              "original_width": 1200, "original_height": 300, "image_rotation": 0,
              "value": {"points": [[25, 10], [75, 10], [75, 90]], "polygonlabels": ["treeGroup"]}}
    return {"project": 3, "data": {"image": f"/data/local-files/?d=classroom_runs%2Fclass-20261108%2Fimages%2F{name}"},
            "annotations": [{"id": 1, "updated_at": "2026-11-08T10:00:00Z", "result": [region] if polygons else []}]}


def test_annotation_import_shared_image_and_empty_submission(classroom, tmp_path):
    _, _, plan, state, baseline = classroom
    dimensions = {r["filename"]: (1200, 300) for r in baseline["images"]}
    annotated, empty, missing = task(), task("image001.JPG", False), task("image002.JPG")
    missing["annotations"][0]["was_cancelled"] = True
    # Model predictions inside a task are deliberately ignored.
    missing["predictions"] = [{"result": annotated["annotations"][0]["result"]}]
    write(tmp_path / "s001-annotations.json", [annotated, empty, missing])
    other = task(polygons=False)
    other["project"] = state["students"]["s041"]["project_id"]
    write(tmp_path / "s041-annotations.json", [other])
    imported = import_annotations(tmp_path, plan, state, dimensions)
    assert imported["s001"]["image000.JPG"][0]["polygon"] == [[300, 30], [900, 30], [900, 270]]
    assert imported["s001"]["image000.JPG"][0]["class"] == "treeGroup"
    assert imported["s001"]["image001.JPG"] == []
    assert "image002.JPG" not in imported["s001"]
    assert imported["s041"]["image000.JPG"] == []
    assert imported["s002"] == {}
    assert "Anotováno bez polygonů" in panel("image001.JPG", baseline["images"][1], "student", [])
    assert "Zatím neanotováno" in panel("image002.JPG", baseline["images"][2], "student", None)


@pytest.mark.parametrize("problem", ["project", "unassigned", "duplicate", "run", "rotation", "dimensions", "label", "points", "unknown-student", "checksum"])
def test_reject_bad_exports(classroom, tmp_path, problem):
    _, _, plan, state, baseline = classroom
    dimensions = {r["filename"]: (1200, 300) for r in baseline["images"]}
    item = task()
    region = item["annotations"][0]["result"][0]
    items = [item]
    if problem == "project": item["project"] = 99
    elif problem == "unassigned": item["data"]["image"] = "image050.JPG"
    elif problem == "duplicate": items.append(deepcopy(item))
    elif problem == "run": item["data"]["image"] = item["data"]["image"].replace("class-20261108", "other")
    elif problem == "rotation": region["image_rotation"] = 90
    elif problem == "dimensions": region["original_width"] = 640
    elif problem == "label": region["value"]["polygonlabels"] = ["TreeGroup"]
    elif problem == "points": region["value"]["points"][0][0] = 101
    elif problem == "unknown-student": write(tmp_path / "s999-annotations.json", [])
    elif problem == "checksum": write(tmp_path / "complete.json", {"run_id": plan["run_id"], "sha256": {"s001-annotations.json": "wrong"}})
    write(tmp_path / "s001-annotations.json", items)
    with pytest.raises(ViewerError):
        import_annotations(tmp_path, plan, state, dimensions)


def test_future_finetuned_geometry_and_model_identity(classroom):
    _, images, plan, state, baseline = classroom
    fine = deepcopy(baseline)
    fine["stage"] = "finetuned"
    fine["model"]["id"] = "sha256:new"
    for record in fine["images"]:
        for prediction in record["predictions"]:
            prediction["model_checkpoint_id"] = "sha256:new"
    base, index, _ = validate_inputs(plan, state, baseline, images, fine)
    name = "image000.JPG"
    markup = panel(name, base[name], "finetuned", index[name]["predictions"])
    assert 'class="panel finetuned"' in markup and "90.0%" in markup
    assert "Model zatím nebyl doučen" not in markup


@pytest.mark.parametrize("bad", ["../x.JPG", "a/../x.JPG", "a%2F..%2Fx.JPG", "file.JPG:stream", "a%00.JPG"])
def test_task_path_safety(bad):
    with pytest.raises(ViewerError): task_filename(bad)


def test_output_path_safety_and_html_escaping(classroom, tmp_path):
    root, images, plan, state, baseline = classroom
    for output in (images, images / "new", root / "plan.json", root):
        with pytest.raises(ViewerError): protect_output(output, arguments(classroom, output))
    local_plan = deepcopy(plan)
    local_plan["students"][0]["name"] = '<script>alert("x")</script>'
    write(tmp_path / "plan.json", local_plan)
    args = arguments(classroom, tmp_path / "viewer", student="s001")
    args.plan = tmp_path / "plan.json"
    build(args)
    page = (args.output / "s001.html").read_text(encoding="utf8")
    assert '<script>alert' not in page and '&lt;script&gt;' in page


def test_latest_submitted_annotation_and_archive_checksum(classroom, tmp_path):
    _, _, plan, state, baseline = classroom
    dimensions = {r["filename"]: (1200, 300) for r in baseline["images"]}
    item = task()
    later = {"id": 2, "updated_at": "2026-11-08T11:00:00Z", "result": []}
    item["annotations"].append(later)
    path = tmp_path / "s001-annotations.json"
    write(path, [item])
    write(tmp_path / "complete.json", {"run_id": plan["run_id"], "sha256": {path.name: digest(path)}})
    assert import_annotations(tmp_path, plan, state, dimensions)["s001"]["image000.JPG"] == []


def test_regenerate_with_exports_and_finetuned_without_inference(classroom, tmp_path):
    _, _, plan, state, baseline = classroom
    archive = tmp_path / "archive"
    archive.mkdir()
    write(archive / "s001-annotations.json", [task(), task("image001.JPG", False)])
    # Available export, but no submitted annotations yet.
    write(archive / "s002-annotations.json", [])
    fine = deepcopy(baseline)
    fine["stage"] = "finetuned"
    fine_path = tmp_path / "finetuned.json"
    write(fine_path, fine)
    args = arguments(classroom, tmp_path / "viewer", annotations=archive, finetuned=fine_path)
    report = build(args)
    assert report["submitted_tasks"]["s001"] == 2
    page = (args.output / "s001.html").read_text(encoding="utf8")
    assert "2 / 5 anotováno" in page and "Anotováno bez polygonů" in page
    assert "Model zatím nebyl doučen" not in page
    assert "0 / 5 anotováno" in (args.output / "s002.html").read_text(encoding="utf8")
    assert "Průběh zatím není k dispozici" in (args.output / "index.html").read_text(encoding="utf8")
    svgs = [ET.fromstring(x) for x in re.findall(r"<svg\b.*?</svg>", page, flags=re.S)]
    assert svgs[0].find("g/polygon").attrib["points"].startswith("300.000000,30.000000")
    assert svgs[2].find("g/polygon") is not None
