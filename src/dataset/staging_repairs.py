"""Deterministic zero-length-edge removal, exclusively in v5 staging."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from .reproducible import sha256
from .review import geometry

REPAIR_STATUS='AUTO_REPAIRED_DUPLICATE_VERTEX'


def remove_consecutive(points):
    """Return points and zero-based original indices; never remove a closing point.

    Equality is exact, not a tolerance. The nonzero cyclic edge sequence must
    remain identical, which proves boundary preservation without area heuristics.
    """
    kept=[]; removed=[]
    for index,point in enumerate(points):
        point=list(point)
        if kept and point==kept[-1]: removed.append(index)
        else: kept.append(point)
    def edges(ring):
        return [(a,b) for a,b in zip(ring,ring[1:]+ring[:1]) if a!=b]
    original=[list(p) for p in points]
    if edges(original)!=edges(kept): raise ValueError('Duplicate removal changed boundary')
    if removed and geometry(kept)['invalid']:
        raise ValueError('Polygon has an independent issue; not a safe duplicate-only repair')
    return kept,removed


def generated_points(points,width,height):
    normalized=[(min(max(x,0),width),min(max(y,0),height)) if -1<=x<=width+1 and -1<=y<=height+1 else (x,y) for x,y in points]
    return [[round(x/width,8)*width,round(y/height,8)*height] for x,y in normalized]


def immutable_write(path,data):
    encoded=(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    if path.exists():
        if path.read_bytes()!=encoded: raise ValueError('Refusing to overwrite changed staging artifact: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(encoded)


def repair_cases(project,catalog,records):
    project=Path(project).resolve(); root=project/'data_staging/benchmark_v5/auto_duplicate_repairs'
    if root.resolve()!=root: raise ValueError('Unsafe staging junction')
    by_annotation={r['source_annotation_path']:r for r in records}
    groups={}
    for row in catalog:
        if row['issue_type']=='GEOMETRY' and row['recommended_action']=='SAFE_REMOVE_DUPLICATE_VERTEX':
            groups.setdefault(row['annotation_a'],[]).append(row)
    manifest=[]
    for source,rows in sorted(groups.items()):
        record=by_annotation[source]; digest=sha256(source)
        if digest!=record['annotation_sha256'] or any(r['annotation_a_sha256']!=digest for r in rows): raise ValueError('Stale repair source')
        original=json.loads(Path(source).read_text(encoding='utf-8-sig'))
        raw=copy.deepcopy(original); events=[]
        for row in rows:
            index=json.loads(row['geometry_issue'])['shape_index']-1
            before=raw['shapes'][index]['points']; after,removed=remove_consecutive(before)
            if removed:
                raw['shapes'][index]['points']=after
                events.append(dict(review_id=row['review_id'],shape_index=index+1,repair_type='RAW_CONSECUTIVE_DUPLICATE',coordinate_space='raw_pixels',original_vertex_count=len(before),repaired_vertex_count=len(after),removed_vertex_indices=removed,index_base=0,boundary_preserved=True))
        exported=copy.deepcopy(raw)
        for shape in exported['shapes']:
            if shape.get('shape_type','polygon').casefold()=='polygon':
                shape['points']=generated_points(shape['points'],record['width'],record['height'])
        for row in rows:
            index=json.loads(row['geometry_issue'])['shape_index']-1
            before=exported['shapes'][index]['points']; after,removed=remove_consecutive(before)
            if removed:
                exported['shapes'][index]['points']=after
                events.append(dict(review_id=row['review_id'],shape_index=index+1,repair_type='EXPORT_ROUNDING_CONSECUTIVE_DUPLICATE',coordinate_space='generated_pixels_after_8_decimal_normalized_rounding',original_vertex_count=len(before),repaired_vertex_count=len(after),removed_vertex_indices=removed,index_base=0,boundary_preserved=True))
        if {e['review_id'] for e in events}!={r['review_id'] for r in rows}: raise ValueError('Not every duplicate case was repaired')
        directory=root/digest
        raw_path=directory/'annotation.json'; export_path=directory/'generated_annotation.json'
        immutable_write(raw_path,raw); immutable_write(export_path,exported)
        entry=dict(original_annotation=source,original_annotation_sha256=digest,source_image=record['source_image_path'],content_sha256=record['content_sha256'],staged_annotation=str(raw_path),staged_annotation_sha256=sha256(raw_path),generated_annotation=str(export_path),generated_annotation_sha256=sha256(export_path),status=REPAIR_STATUS,repairs=events,export_precision=8,generator_sha256=sha256(Path(__file__)))
        for path in (raw_path,export_path):
            immutable_write(path.with_suffix('.provenance.json'),entry|{'fixed_annotation_sha256':sha256(path),'artifact_coordinate_space':'raw' if path==raw_path else 'generated'})
        if sha256(source)!=digest: raise ValueError('Raw source changed during staging')
        manifest.append(entry)
    immutable_write(root/'manifest.json',manifest)
    return manifest
