"""Decision-gated v5 build. All preparation completes before replacing an output."""
import csv
import hashlib
import json
import random
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path

from PIL import Image
import yaml
from .reproducible import sha256,write_json,write_csv,validate_output
from .coco_export import polygon_to_coco,dataset_fingerprint
from .staging_repairs import immutable_write
from .review import geometry
from .final_decisions import validate_decisions,print_report,annotation_plan,public_report


def move_completed(source,target):
    # Windows scanners can briefly retain handles after validation closes files.
    for delay in (0.05,0.1,0.2,0.4,0.8):
        try: return source.rename(target)
        except PermissionError: time.sleep(delay)
    return source.rename(target)


def grouped_split(records):
    groups=defaultdict(list)
    for row in records:
        # Conservatively group camera frame names globally, across annotators/folders.
        group=re.sub(r'_\d+$','',Path(row['source_image_path']).stem).casefold()
        groups[group].append(row)
    if len(groups)<3: raise ValueError('At least three parent-image groups are needed for train/val/test')
    items=sorted(groups.items()); random.Random(42).shuffle(items);items.sort(key=lambda x:-len(x[1]))
    targets=dict(train=.7*len(records),val=.15*len(records),test=.15*len(records));counts=dict.fromkeys(targets,0);assignments={}
    for i,(group,members) in enumerate(items):
        empty=[s for s in counts if counts[s]==0]
        choices=empty if len(items)-i==len(empty) else list(counts)
        split=min(choices,key=lambda s:(counts[s]+len(members)-targets[s])**2-(counts[s]-targets[s])**2)
        counts[split]+=len(members)
        for row in members: assignments[row['content_sha256']]=(split,group)
    return assignments,counts


def build_reviewed(project,rebuild=False):
    project=Path(project).resolve(); result=validate_decisions(project)
    strict=yaml.safe_load((project/'config/dataset_sources.yaml').read_text(encoding='utf-8')).get('benchmark_v5_strict_geometry',False)
    print_report(result)
    if not result['safe_to_build']: raise ValueError('benchmark_v5 blocked: complete and validate all required decisions first')
    out=project/'datasets/benchmark_v5';staging=project/'data_staging/benchmark_v5';raw=Path(result['_audit']['raw_root']).resolve()
    for target in (out,staging):
        if target.resolve()!=target or not target.is_relative_to(project) or target.is_relative_to(raw) or raw.is_relative_to(target): raise ValueError('Unsafe v5 output path')
    if out.exists() and not rebuild: raise ValueError('benchmark_v5 exists; pass --rebuild')
    grouped=defaultdict(list)
    for record in result['_records']: grouped[record['content_sha256']].append(record)
    plans=[]
    for content,members in sorted(grouped.items()):
        plan=result['_plans'].get(content)
        if plan and plan.get('exclude'): continue
        if not plan:
            safe=[r for r in result['_catalog'] if r['content_sha256']==content and r.get('review_status')=='AUTO_REPAIRED_DUPLICATE_VERTEX']
            candidates=[r for r in members if r['inclusion_status'] in {'included','duplicate'}]
            if not candidates and safe: candidates=[r for r in members if r['source_annotation_path']==safe[0]['annotation_a']]
            if not candidates: continue  # Existing explicit label/shape exclusions.
            record=sorted(candidates,key=lambda r:r['source_annotation_path'])[0]
            plan=annotation_plan(project,record['source_annotation_path'],record,[],{},[s for s in safe if s['annotation_a']==record['source_annotation_path']])|{'record':record}
        if strict and (plan['allowed_self_intersections'] or any(geometry(s['points'])['invalid'] for s in plan['data']['shapes'])):
            raise ValueError('Strict geometry preflight rejected an invalid selected annotation: '+plan['original_annotation'])
        plans.append(plan)
    selected=[p['record'] for p in plans]
    if len({r['content_sha256'][:16] for r in selected})!=len(selected): raise ValueError('Staging filename hash-prefix collision')
    assignments,counts=grouped_split(selected)
    pixels={}
    for record in selected:
        with Image.open(record['source_image_path']) as im:
            im=im.convert('RGB'); digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        if digest in pixels: raise ValueError('Identical decoded image content remains after deduplication')
        pixels[digest]=record['content_sha256'];record['pixel_sha256']=digest
    # All approval/geometry checks above were read-only. Stage only after success.
    recipe=hashlib.sha256(json.dumps([p['data'] for p in plans],sort_keys=True).encode()).hexdigest()
    generation=staging/'approved_generations'/(result['decision_file_sha256'][:16]+'_'+recipe[:16])
    if generation.resolve()!=generation: raise ValueError('Unsafe staging generation path')
    generation.mkdir(parents=True,exist_ok=True)
    work=staging/('build_'+uuid.uuid4().hex[:8]);work.mkdir()
    canonical=project/'config/benchmark_v5_decisions.csv'
    shutil.copy2(canonical,work/'decisions.csv')
    coco={s:dict(images=[],annotations=[],categories=[{'id':1,'name':'tree'},{'id':2,'name':'TreeGroup'}]) for s in counts}
    rows=[];allowed={}
    for global_image_id,plan in enumerate(sorted(plans,key=lambda p:p['record']['content_sha256']),1):
        record=plan['record'];identity=record['content_sha256'];split,parent=assignments[identity]
        filename=identity+Path(record['source_image_path']).suffix.lower();w,h=record['width'],record['height']
        annotation_path=generation/'annotations'/f'{identity[:16]}.json'
        if annotation_path.resolve()!=annotation_path: raise ValueError('Unsafe staged annotation path')
        immutable_write(annotation_path,plan['data'])
        annotation_hash=sha256(annotation_path)
        immutable_write(annotation_path.with_suffix('.provenance.json'),dict(original_annotation=plan['original_annotation'],original_annotation_sha256=plan['original_annotation_sha256'],fixed_annotation_sha256=annotation_hash,content_sha256=identity,decision_file_sha256=result['decision_file_sha256'],repairs=plan['provenance'],allowed_self_intersections=plan['allowed_self_intersections']))
        image_path=work/split/'images'/filename;image_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(record['source_image_path'],image_path)
        if sha256(image_path)!=identity: raise ValueError('Source image changed during build')
        dataset=coco[split];iid=global_image_id
        dataset['images'].append(dict(id=iid,file_name=filename,width=w,height=h,canonical_image_id=identity))
        labels=[];classes=[]
        for shape in plan['data']['shapes']:
            cid=0 if shape['label'].strip().casefold()=='tree' else 1;classes.append(cid)
            values=[f'{v:.8f}' for x,y in shape['points'] for v in (x/w,y/h)]
            labels.append(str(cid)+' '+' '.join(values))
            points=[(float(values[i])*w,float(values[i+1])*h) for i in range(0,len(values),2)]
            seg,box,area=polygon_to_coco(points,w,h)
            dataset['annotations'].append(dict(id=len(dataset['annotations'])+1,image_id=iid,category_id=cid+1,segmentation=[seg],bbox=box,area=area,iscrowd=0))
        label=work/split/'labels'/f'{identity}.txt';label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(labels),encoding='utf-8')
        allowed[identity]=plan['allowed_self_intersections']
        rows.append(dict(image_id=identity,filename=filename,content_sha256=identity,pixel_sha256=record['pixel_sha256'],annotator=record['source_annotator'],source_dataset=record['source_dataset'],source_image_path=record['source_image_path'],source_annotation_path=record['source_annotation_path'],source_image=record['source_image_path'],source_annotation=record['source_annotation_path'],original_annotation_sha256=plan['original_annotation_sha256'],staged_annotation=str(annotation_path),annotation_sha256=annotation_hash,split=split,parent_image_group=parent,width=w,height=h,class_tree_count=classes.count(0),class_treegroup_count=classes.count(1),total_instances=len(classes),number_of_objects=len(classes)))
    write_csv(work/'dataset_manifest.csv',rows);fingerprint=dataset_fingerprint(work/'dataset_manifest.csv')
    for split,data in coco.items():
        (work/split/'images').mkdir(parents=True,exist_ok=True);(work/split/'labels').mkdir(exist_ok=True)
        data['info']={'dataset_fingerprint':fingerprint}
        write_json(work/'coco/annotations'/f'instances_{split}.json',data)
    (work/'coco/dataset_fingerprint.txt').write_text(fingerprint+'\n')
    (work/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='train/images',val='val/images',test='test/images',names={0:'tree',1:'TreeGroup'})),encoding='utf-8')
    validation=validate_output(work,len(rows),None if strict else allowed)
    by_parent=defaultdict(set)
    for row in rows: by_parent[row['parent_image_group']].add(row['split'])
    if any(len(s)>1 for s in by_parent.values()): raise ValueError('Parent-image leakage')
    validation.update(content_leakage=False,decoded_pixel_duplicates=False,parent_image_group_leakage=False,spatial_leakage_status='PARENT_FILENAME_GROUPED; FLIGHT/SPATIAL METADATA UNVERIFIED')
    write_json(work/'validation.json',validation)
    summary=dict(decision_file=str(canonical),decision_file_sha256=result['decision_file_sha256'],review_catalog_sha256=sha256(project/'config/benchmark_v5_review_catalog.json'),source_snapshot_sha256=sha256(project/'config/benchmark_v5_source_snapshot.json'),generator_sha256=sha256(Path(__file__)),decision_validator_sha256=sha256(Path(__file__).with_name('final_decisions.py')),strict_geometry_policy=strict,random_seed=42,split_method='global parent-frame filename groups; deterministic greedy 70/15/15 targets',images_per_split=counts,included_image_count=len(rows),explicit_self_intersection_exceptions={k:v for k,v in allowed.items() if v},spatial_leakage_status=validation['spatial_leakage_status'],dataset_fingerprint=fingerprint)
    write_json(work/'dataset_summary.json',summary)
    # Re-read canonical decisions and every raw source immediately before publication.
    final=validate_decisions(project)
    if not final['safe_to_build'] or final['decision_file_sha256']!=result['decision_file_sha256']: raise ValueError('Inputs changed during build')
    for plan in plans:
        for event in plan['provenance']:
            if event['repair_type']=='HUMAN_MANUAL_COPY' and sha256(event['path'])!=event['provenance']['fixed_annotation_sha256']: raise ValueError('Manual copy changed during build')
    out.parent.mkdir(parents=True,exist_ok=True)
    backup=None
    if out.exists():
        backup=out.with_name('benchmark_v5_previous_'+uuid.uuid4().hex)
        if backup.resolve()!=backup: raise ValueError('Unsafe backup path')
        move_completed(out,backup)
    try: move_completed(work,out)
    except OSError:
        if backup: move_completed(backup,out)
        raise
    return summary
