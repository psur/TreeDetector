"""Reproducible, read-only LabelMe source audit and guarded benchmark generation."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import yaml

from .annotations import parse_labelme
from .coco_export import polygon_to_coco
from .scanner import EXTS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ring_key(points):
    points = [tuple(p) for p in points]
    if points and points[0] == points[-1]:
        points.pop()
    if not points:
        return ()
    return min(tuple(p[i:] + p[:i]) for p in (points, list(reversed(points))) for i in range(len(p)))


def valid_polygon(points, width, height):
    polygon_to_coco(points, width, height)
    points = list(ring_key(points))
    if len(set(points)) != len(points):
        raise ValueError('repeated polygon vertex')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a, b, c):
        return cross(a,b,c) == 0 and min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a,b) in enumerate(edges):
        for j in range(i+2, len(edges)):
            if i == 0 and j == len(edges)-1:
                continue
            c,d = edges[j]
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                raise ValueError('self-intersecting polygon')


def split_counts(n):
    train = int(n * .70)
    val = int(n * .15)
    return {'train': train, 'val': val, 'test': n-train-val}


def audit_sources(root, rules):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    folders, records, parsed = [], [], {}
    groups = defaultdict(list)
    files = sorted(root.rglob('*'), key=lambda p: p.as_posix().casefold())
    for directory in [root] + [p for p in files if p.is_dir()]:
        children = [p for p in directory.iterdir() if p.is_file()]
        images = sorted([p for p in children if p.suffix.lower() in EXTS])
        annotations = sorted([p for p in children if p.suffix.lower() == '.json'])
        imap, amap = defaultdict(list), defaultdict(list)
        for p in images: imap[p.stem.casefold()].append(p)
        for p in annotations: amap[p.stem.casefold()].append(p)
        rel = directory.relative_to(root).as_posix()
        annotator = directory.relative_to(root).parts[0] if directory != root else ''
        folder = dict(annotator=annotator, folder=rel, images=len(images), json=len(annotations), matched_pairs=0,
                      orphan_images=[], orphan_json=[], invalid_json=[], labels={}, shape_types={}, problems=[])
        data_by_path = {}
        labels, kinds = Counter(), Counter()
        for path in annotations:
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(data, dict) or not isinstance(data.get('shapes'), list) or 'imagePath' not in data:
                    raise ValueError('not a LabelMe image+JSON annotation')
                for shape in data['shapes']:
                    labels[str(shape.get('label','')).strip()] += 1
                    kinds[str(shape.get('shape_type','polygon')).strip()] += 1
                data_by_path[path] = data
            except (ValueError, TypeError, AttributeError, OSError) as exc:
                folder['invalid_json'].append({'path': str(path), 'reason': str(exc)})
            if path.stem.casefold() not in imap:
                folder['orphan_json'].append(str(path))
        folder['labels'], folder['shape_types'] = dict(labels), dict(kinds)
        for image in images:
            matches = amap[image.stem.casefold()]
            reasons = []
            path = matches[0] if len(matches) == 1 else None
            if not matches:
                folder['orphan_images'].append(str(image)); reasons.append('missing annotation')
            elif len(matches) != 1 or len(imap[image.stem.casefold()]) != 1:
                reasons.append('ambiguous image/annotation stem')
            else:
                folder['matched_pairs'] += 1
            record = dict(image_id='', source_annotator=annotator, source_dataset=rel,
                          source_image_path=str(image), source_annotation_path=str(path or ''), content_sha256='',
                          annotation_sha256='', labels=[], shape_types=[], annotation_count=0,
                          inclusion_status='excluded', exclusion_reason='', duplicate_group='', width=0, height=0)
            signature = geometry = None
            try:
                record['content_sha256'] = record['image_id'] = sha256(image)
                with Image.open(image) as im: im.verify()
                with Image.open(image) as im: width, height = im.size
                record.update(width=width, height=height)
                if path:
                    record['annotation_sha256'] = sha256(path)
                    data = data_by_path.get(path)
                    if data is None:
                        reasons.append('invalid JSON or unsupported annotation structure')
                    else:
                        shapes = data['shapes']
                        record['labels'] = sorted({str(s.get('label','')).strip() for s in shapes})
                        record['shape_types'] = sorted({str(s.get('shape_type','polygon')).strip() for s in shapes})
                        record['annotation_count'] = len(shapes)
                        # Compare all variants, including excluded variants, before selecting any source.
                        normalized = [(str(s.get('label','')).strip().casefold(), str(s.get('shape_type','polygon')).strip().casefold(), ring_key(s.get('points',[])), s.get('group_id'), s.get('flags',{})) for s in shapes]
                        signature = json.dumps(sorted(normalized, key=repr), sort_keys=True)
                        geometry = repr(sorted((x[1], x[2]) for x in normalized))
                        for label in record['labels']:
                            if label.casefold() in {s.casefold() for s in rules['exclude_labels']} or label.casefold() not in {'tree','treegroup'}:
                                reasons.append('unsupported/excluded label: '+label)
                        for kind in record['shape_types']:
                            if kind.casefold() != 'polygon' or kind.casefold() in rules['exclude_shape_types']:
                                reasons.append('unsupported/excluded shape type: '+kind)
                        if data.get('imageWidth') != width or data.get('imageHeight') != height:
                            reasons.append('annotation dimensions do not match image')
                        if not reasons:
                            annotation = parse_labelme(path, image, width, height, {0:'tree',1:'TreeGroup'})
                            for obj in annotation.objects:
                                valid_polygon(obj.polygon, width, height)
                                # Validate the geometry that will actually be serialized to YOLO.
                                valid_polygon([(round(x/width,8)*width,round(y/height,8)*height) for x,y in obj.polygon],width,height)
                            parsed[str(image)] = annotation
            except (ValueError, TypeError, IndexError, KeyError, OSError) as exc:
                reasons.append(str(exc))
            record['exclusion_reason'] = '; '.join(sorted(set(reasons)))
            if reasons:
                folder['problems'].append({'image': str(image), 'reason': record['exclusion_reason']})
            if not reasons: record['inclusion_status'] = 'eligible'
            records.append(record)
            if record['content_sha256']:
                groups[record['content_sha256']].append((record,signature,geometry))
        folders.append(folder)
    eligible_before = sum(r['inclusion_status']=='eligible' for r in records)
    duplicates, conflicts = [], []
    for digest, members in sorted(groups.items()):
        if len(members) < 2: continue
        for record, _, _ in members: record['duplicate_group'] = digest
        details = [dict(source_image_path=r['source_image_path'], source_annotation_path=r['source_annotation_path'],
                        annotator=r['source_annotator'], annotation_count=r['annotation_count'], labels=r['labels'],
                        exclusion_reason=r['exclusion_reason']) for r,_,_ in members]
        entry = dict(content_sha256=digest, sources=details,
                     within_annotator=len({r['source_annotator'] for r,_,_ in members})<len(members),
                     between_annotators=len({r['source_annotator'] for r,_,_ in members})>1,
                     between_folders=len({r['source_dataset'] for r,_,_ in members})>1)
        duplicates.append(entry)
        signatures = {s for _,s,_ in members if s is not None}
        if len(signatures)>1 or (any(s is None and r['source_annotation_path'] for r,s,_ in members) and any(r['inclusion_status']=='eligible' for r,_,_ in members)):
            conflicts.append(entry | {'polygon_geometry_differs':len({g for _,_,g in members})>1})
            for r,_,_ in members:
                if r['inclusion_status']=='eligible':
                    r['inclusion_status']='conflict'; r['exclusion_reason']='unresolved duplicate annotation conflict'
        else:
            candidates = [r for r,_,_ in members if r['inclusion_status']=='eligible']
            for r in candidates[1:]:
                r['inclusion_status']='duplicate'; r['exclusion_reason']='identical annotation duplicate; canonical source: '+candidates[0]['source_image_path']
    selected = [r for r in records if r['inclusion_status']=='eligible']
    for r in selected: r['inclusion_status']='included'
    for f in folders:
        f['duplicate_images'] = sum(bool(r['duplicate_group']) for r in records if r['source_dataset']==f['folder'])
    tile_groups = defaultdict(list)
    for r in records:
        match = re.match(r'(.+)_\d+$',Path(r['source_image_path']).stem)
        if match: tile_groups[r['source_dataset']+'/'+match[1]].append(r['source_image_path'])
    report = dict(raw_root=str(root), folders=folders, total_matched_pairs=sum(f['matched_pairs'] for f in folders),
                  eligible_before_deduplication=eligible_before, duplicate_groups=duplicates, annotation_conflicts=conflicts,
                  excluded_image_count=sum(r['inclusion_status']=='excluded' for r in records),
                  duplicate_count=sum(r['inclusion_status']=='duplicate' for r in records),
                  expected_unique_images=None if conflicts else len(selected), uncontested_unique_images=len(selected), eligible_content_groups=len({r['content_sha256'] for r in records if r['inclusion_status'] in {'included','duplicate','conflict'}}),
                  recommended_split=None if conflicts else split_counts(len(selected)), safe_to_build=not conflicts and len(selected)>=3,
                  spatial_leakage_status='UNVERIFIED', candidate_filename_groups={k:v for k,v in tile_groups.items() if len(v)>1},
                  grouping_recommendation='Filenames suggest tiles from larger images. Confirm parent image/flight identities before a grouped split; names alone do not establish spatial independence.')
    return report, records, parsed


def validate_output(out, expected, allowed_self_intersections=None):
    rows = list(csv.DictReader((out/'dataset_manifest.csv').open(encoding='utf-8')))
    if len(rows)!=expected or len({r['image_id'] for r in rows})!=expected or len({r['content_sha256'] for r in rows})!=expected:
        raise ValueError('duplicate IDs/content or incorrect image total')
    if any(r['split'] not in {'train','val','test'} for r in rows):
        raise ValueError('invalid manifest split')
    total = 0
    for split in ('train','val','test'):
        data=json.loads((out/'coco'/'annotations'/f'instances_{split}.json').read_text(encoding='utf-8'))
        subset={r['image_id']:r for r in rows if r['split']==split}
        images={im['id']:im for im in data['images']}
        if len(images)!=len(data['images']) or {im['canonical_image_id'] for im in images.values()}!=set(subset):
            raise ValueError('COCO/manifest image mismatch')
        if {c['id'] for c in data['categories']}!={1,2}: raise ValueError('invalid COCO categories')
        if len({a['id'] for a in data['annotations']})!=len(data['annotations']): raise ValueError('duplicate annotation ID')
        grouped=defaultdict(list)
        for a in data['annotations']:
            if a['image_id'] not in images or a['category_id'] not in {1,2}: raise ValueError('invalid COCO reference')
            grouped[a['image_id']].append(a)
        for iid,im in images.items():
            r=subset[im['canonical_image_id']]
            if sha256(out/split/'images'/r['filename'])!=r['content_sha256']: raise ValueError('image hash mismatch')
            lines=(out/split/'labels'/f"{r['image_id']}.txt").read_text().splitlines()
            if len(lines)!=int(r['total_instances']) or len(lines)!=len(grouped[iid]): raise ValueError('annotation count mismatch')
            for polygon_index,(line,a) in enumerate(zip(lines,grouped[iid]),1):
                cells=line.split(); cid=int(cells[0]); coords=list(map(float,cells[1:]))
                if cid not in {0,1} or len(coords)<6 or len(coords)%2 or not all(0<=v<=1 for v in coords): raise ValueError('invalid YOLO row')
                points=[(coords[i]*im['width'],coords[i+1]*im['height']) for i in range(0,len(coords),2)]
                try: valid_polygon(points,im['width'],im['height'])
                except ValueError as exc:
                    if str(exc)!='self-intersecting polygon' or polygon_index not in (allowed_self_intersections or {}).get(r['image_id'],[]): raise
                seg,box,area=polygon_to_coco(points,im['width'],im['height'])
                if a['category_id']!=cid+1 or a['segmentation']!=[seg] or a['bbox']!=box or a['area']!=area: raise ValueError('COCO/YOLO geometry mismatch')
            total += len(lines)
    return {'valid':True,'images':len(rows),'instances':total,'spatial_leakage_status':'UNVERIFIED'}


def prepare_reproducible(cfg, dataset, rebuild=False, audit_only=False, decisions=None):
    if dataset != 'benchmark_v5': raise ValueError('This workflow only writes benchmark_v5; benchmark_v4 is frozen')
    project=Path(cfg['_project_root']).resolve()
    if ((project/'config/benchmark_v5_review_catalog.json').exists() or (project/'config/benchmark_v5_decisions.csv').exists()) and not audit_only:
        if decisions: raise ValueError('Builds require config/benchmark_v5_decisions.csv; finalize review edits first')
        from .reviewed_build import build_reviewed
        return build_reviewed(project,rebuild)
    definitions=yaml.safe_load((project/'config'/'dataset_sources.yaml').read_text(encoding='utf-8'))
    raw=Path(definitions['raw_root']).resolve()
    out=project/'datasets'/dataset; staging=project/'data_staging'/dataset; reports=project/'results'/f'{dataset}_audit'
    # Resolve all targets before any write; never follow a generated-directory junction into raw/v4.
    for target in (out,staging,reports):
        resolved=target.resolve()
        if resolved!=target or not resolved.is_relative_to(project) or resolved.is_relative_to(raw) or raw.is_relative_to(resolved):
            raise ValueError(f'Unsafe generated output path: {target}')
    if not audit_only and (out.exists() or staging.exists()) and not rebuild:
        raise ValueError('Generated v5 output already exists; pass --rebuild explicitly')
    report, records, parsed=audit_sources(raw,definitions)
    decision_path=Path(decisions) if decisions else project/'config/benchmark_v5_decisions.csv'
    if not decision_path.is_absolute(): decision_path=project/decision_path
    if decisions or decision_path.exists():
        if (project/'config/benchmark_v5_review_catalog.json').exists():
            from .final_decisions import validate_decisions,public_report
            report['human_decision_plan']=public_report(validate_decisions(project,decision_path))
        else:
            from .review_decisions import read_decisions
            report['human_decision_plan']=read_decisions(decision_path,project/'results/benchmark_v5_review/review_catalog.json',records,project)
        # Reading a decision file is not authorization to publish a benchmark.
        report['safe_to_build']=False
    if definitions.get('benchmark_v5_build_blocked',False):
        report['safe_to_build']=False
    reports.mkdir(parents=True,exist_ok=True)
    write_json(reports/'audit.json',report); write_json(reports/'annotation_conflicts.json',report['annotation_conflicts'])
    write_json(reports/'provenance.json',records)
    (reports/'audit_summary.md').write_text(format_audit(report, records), encoding='utf-8')
    print(format_audit(report, records))
    print(json.dumps({k:v for k,v in report.items() if k not in {'folders','duplicate_groups','annotation_conflicts','candidate_filename_groups'}} | {'duplicate_groups':len(report['duplicate_groups']),'annotation_conflicts':len(report['annotation_conflicts'])},indent=2))
    if audit_only: return report
    if not report['safe_to_build']: raise ValueError(f'Unsafe to build benchmark_v5. Review {reports / "annotation_conflicts.json"}')
    # Only exact, resolved generated targets may be replaced after a successful audit.
    for target in (out,staging):
        if target.exists(): shutil.rmtree(target)
        target.mkdir(parents=True)
    write_json(staging/'provenance.json',records)
    chosen=sorted((r for r in records if r['inclusion_status']=='included'),key=lambda r:r['image_id'])
    random.Random(42).shuffle(chosen)
    counts=split_counts(len(chosen)); assignments=['train']*counts['train']+['val']*counts['val']+['test']*counts['test']
    rows=[]; coco={s:dict(images=[],annotations=[],categories=[{'id':1,'name':'tree'},{'id':2,'name':'TreeGroup'}]) for s in counts}
    historical=defaultdict(list)
    v4=project/'datasets'/'benchmark_v4'/'dataset_manifest.csv'
    if v4.exists():
        with v4.open(encoding='utf-8') as stream:
            for old in csv.DictReader(stream):
                image=project/'datasets'/'benchmark_v4'/old['split']/'images'/(old['image_id']+Path(old['source_image']).suffix.lower())
                if image.exists(): historical[sha256(image)].append({'image_id':old['image_id'],'split':old['split']})
    for r,split in zip(chosen,assignments):
        a=parsed[r['source_image_path']]; identity=r['image_id']; filename=identity+Path(r['source_image_path']).suffix.lower()
        stage_image=staging/'images'/filename; stage_image.parent.mkdir(exist_ok=True)
        stage_json=staging/'annotations'/f'{identity}.json'; stage_json.parent.mkdir(exist_ok=True)
        shutil.copy2(a.image_path,stage_image); shutil.copy2(a.annotation_path,stage_json)
        if sha256(stage_image)!=r['content_sha256'] or sha256(stage_json)!=r['annotation_sha256']: raise ValueError('Raw source changed during generation')
        dest=out/split/'images'/filename; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(stage_image,dest)
        label=out/split/'labels'/f'{identity}.txt'; label.parent.mkdir(parents=True,exist_ok=True)
        data=coco[split]; iid=len(data['images'])+1
        data['images'].append(dict(id=iid,file_name=str(dest),width=a.width,height=a.height,canonical_image_id=identity))
        lines=[]
        for obj in a.objects:
            values=[f'{v:.8f}' for x,y in obj.polygon for v in (x/a.width,y/a.height)]
            lines.append(str(obj.class_id)+' '+' '.join(values))
            points=[(float(values[i])*a.width,float(values[i+1])*a.height) for i in range(0,len(values),2)]
            seg,box,area=polygon_to_coco(points,a.width,a.height)
            data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=iid,category_id=obj.class_id+1,segmentation=[seg],bbox=box,area=area,iscrowd=0))
        label.write_text('\n'.join(lines),encoding='utf-8')
        rows.append(dict(image_id=identity,filename=filename,content_sha256=identity,annotator=r['source_annotator'],source_dataset=r['source_dataset'],source_image_path=str(a.image_path),source_annotation_path=str(a.annotation_path),source_image=str(a.image_path),source_annotation=str(a.annotation_path),split=split,width=a.width,height=a.height,class_tree_count=sum(o.class_id==0 for o in a.objects),class_treegroup_count=sum(o.class_id==1 for o in a.objects),total_instances=len(a.objects),number_of_objects=len(a.objects),annotation_sha256=r['annotation_sha256'],v4_assignments=json.dumps(historical[identity])))
    write_csv(out/'dataset_manifest.csv',sorted(rows,key=lambda r:r['image_id']))
    for split,data in coco.items():
        (out/split/'images').mkdir(parents=True,exist_ok=True); (out/split/'labels').mkdir(exist_ok=True)
        write_json(out/'coco'/'annotations'/f'instances_{split}.json',data)
    (out/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='train/images',val='val/images',test='test/images',names={0:'tree',1:'TreeGroup'})),encoding='utf-8')
    from .coco_export import dataset_fingerprint
    (out/'coco'/'dataset_fingerprint.txt').write_text(dataset_fingerprint(out/'dataset_manifest.csv')+'\n')
    validation=validate_output(out,len(chosen)); write_json(out/'validation.json',validation)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=project,capture_output=True,text=True).stdout.strip()
    summary=dict(generation_timestamp=datetime.now(timezone.utc).isoformat(),git_commit=commit,random_seed=42,split_fractions={'train':.70,'val':.15,'test':.15},source_datasets=sorted({r['source_dataset'] for r in records}),included_image_count=len(chosen),excluded_image_count=report['excluded_image_count'],duplicate_count=report['duplicate_count'],annotation_conflict_count=len(report['annotation_conflicts']),images_per_split=counts,instance_counts={s:len(d['annotations']) for s,d in coco.items()},per_class_counts={'tree':sum(r['class_tree_count'] for r in rows),'TreeGroup':sum(r['class_treegroup_count'] for r in rows)},spatial_leakage_status='UNVERIFIED',source_definitions=definitions,generator_sha256=sha256(Path(__file__)),git_dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=project,capture_output=True,text=True).stdout.strip()))
    write_json(out/'dataset_summary.json',summary)
    return summary


def format_audit(report, records):
    lines = ['RAW SOURCES FOUND:', '| Annotator | Images | JSON | Matched pairs | Orphan JSON | Invalid JSON |', '|---|---:|---:|---:|---:|---:|']
    for annotator in sorted({r['source_annotator'] for r in records}):
        folders = [f for f in report['folders'] if f['annotator']==annotator]
        counts = [sum(f[k] for f in folders) for k in ('images','json','matched_pairs')]
        counts += [sum(len(f[k]) for f in folders) for k in ('orphan_json','invalid_json')]
        lines.append('| '+annotator+' | '+' | '.join(map(str,counts))+' |')
    jakub = [f for f in report['folders'] if f['annotator'].casefold()=='jakub']
    problems = Counter(r['exclusion_reason'] for r in records if r['source_annotator'].casefold()=='jakub' and r['inclusion_status']=='excluded')
    lines += ['', 'JAKUB:']
    for key in ('images','json','matched_pairs'):
        lines.append(key.replace('_',' ')+': '+str(sum(f[key] for f in jakub)))
    lines += ['labels: '+', '.join(sorted({label for f in jakub for label in f['labels']})),
              'shape types: '+', '.join(sorted({kind for f in jakub for kind in f['shape_types']})),
              'problems: '+str(dict(problems))+'; orphan JSON: '+str(sum(len(f['orphan_json']) for f in jakub)),
              '', 'TOTAL MATCHED PAIRS: '+str(report['total_matched_pairs']),
              'ELIGIBLE BEFORE DEDUPLICATION: '+str(report['eligible_before_deduplication']),
              'DUPLICATE GROUPS: '+str(len(report['duplicate_groups'])),
              'ANNOTATION CONFLICTS: '+str(len(report['annotation_conflicts'])),
              'EXCLUDED: '+str(report['excluded_image_count'])+' images; '+str(dict(Counter(r['exclusion_reason'] for r in records if r['inclusion_status']=='excluded'))),
              'EXPECTED UNIQUE IMAGES: '+(str(report['expected_unique_images']) if report['expected_unique_images'] is not None else 'PENDING conflict review; '+str(report['uncontested_unique_images'])+' uncontested, up to '+str(report['eligible_content_groups'])+' eligible unique candidates'),
              'RECOMMENDED SPLIT: 70% / 15% / 15%, seed 42; image-level pending confirmation of parent-image grouping']
    counts = report['recommended_split'] or split_counts(report['eligible_content_groups'])
    for split,count in counts.items():
        lines.append(split+': '+str(count)+(' (conditional on retaining all eligible unique candidates)' if report['annotation_conflicts'] else ''))
    lines += ['SAFE TO BUILD BENCHMARK_V5: '+('YES' if report['safe_to_build'] else 'NO'),
              'SPATIAL LEAKAGE STATUS: UNVERIFIED', '', report['grouping_recommendation'], '',
              'Per-folder counts below are direct children, so nested sources are not double-counted.',
              '| Folder | Images | JSON | Pairs | Orphan images | Orphan JSON | Invalid JSON | Duplicate images | Labels | Shape types |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for f in report['folders']:
        values=[f['folder'],f['images'],f['json'],f['matched_pairs'],len(f['orphan_images']),len(f['orphan_json']),len(f['invalid_json']),f['duplicate_images'],', '.join(f['labels']),', '.join(f['shape_types'])]
        lines.append('| '+' | '.join(map(str,values))+' |')
    return '\n'.join(lines)+'\n'
