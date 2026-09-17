"""Deterministic human source selection followed by conditional geometry decisions."""
import csv
import json
from pathlib import Path

from .annotations import parse_labelme
from .reproducible import sha256, valid_polygon


def read_decisions(path, catalog_path, records, project):
    """Read decisions without building, merging, or repairing self-intersections."""
    project=Path(project).resolve()
    catalog={r['review_id']:r for r in json.loads(Path(catalog_path).read_text(encoding='utf-8'))}
    by_hash={}
    for record in records: by_hash.setdefault(record['content_sha256'],[]).append(record)
    supplied={}
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'review_id','content_sha256','human_decision','selected_annotation'}.issubset(reader.fieldnames or []):
            raise ValueError('Decision CSV missing required columns')
        for row in reader:
            identity=row['review_id']
            if identity not in catalog or identity in supplied: raise ValueError('Unknown or duplicate review_id: '+identity)
            original=catalog[identity]
            for key in ('content_sha256','annotation_a','annotation_b','issue_type'):
                if key in row and row[key]!=str(original[key]): raise ValueError('Decision identity changed: '+identity)
            decision=row['human_decision'].strip(); selected=row['selected_annotation'].strip()
            if not decision and selected: raise ValueError('selected_annotation requires a human_decision')
            if selected and decision in {'KEEP_AS_IS','MANUAL_FIX','AUTO_FIX_APPROVED'}: raise ValueError('Use GEOMETRY_FIXED_COPY to select a repaired annotation')
            valid={'','KEEP_A','KEEP_B','EXCLUDE','GEOMETRY_FIXED_COPY','KEEP_AS_IS','MANUAL_FIX','AUTO_FIX_APPROVED'}
            if decision not in valid and not decision.startswith('KEEP_PATH:'): raise ValueError('Unsupported human decision: '+decision)
            if decision in {'KEEP_AS_IS','MANUAL_FIX','AUTO_FIX_APPROVED'} and original['issue_type']!='GEOMETRY': raise ValueError('Geometry decision requires a geometry review ID')
            supplied[identity]=dict(decision=decision,selected=selected)
            # For a conflict row with linked geometry, these optional explicit
            # decisions avoid adding a duplicate image row to the reduced CSV.
            if row.get('geometry_decisions','').strip():
                linked=json.loads(row['geometry_decisions'])
                if not isinstance(linked,dict): raise ValueError('geometry_decisions must be a JSON object')
                for gid,value in linked.items():
                    if gid not in catalog or catalog[gid]['issue_type']!='GEOMETRY' or catalog[gid]['content_sha256']!=original['content_sha256'] or gid in supplied:
                        raise ValueError('Invalid or duplicate linked geometry decision')
                    if value not in {'KEEP_AS_IS','MANUAL_FIX','EXCLUDE','AUTO_FIX_APPROVED'}: raise ValueError('Unsupported linked geometry decision')
                    supplied[gid]=dict(decision=value,selected='')
    checked=set()
    for original in catalog.values():
        if original['content_sha256'] not in by_hash: raise ValueError('Decision content missing from current audit')
        for side in ('a','b'):
            source=original['annotation_'+side]
            if source and source not in checked:
                if sha256(source)!=original['annotation_'+side+'_sha256']: raise ValueError('Stale annotation: '+source)
                checked.add(source)
        if original['image_path'] not in checked:
            if sha256(original['image_path'])!=original['content_sha256']: raise ValueError('Stale image')
            checked.add(original['image_path'])
    manifest_path=project/'data_staging/benchmark_v5/auto_duplicate_repairs/manifest.json'
    automatic={}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding='utf-8')):
            if sha256(entry['original_annotation'])!=entry['original_annotation_sha256']: raise ValueError('Stale automatic repair source')
            for key in ('staged_annotation','generated_annotation'):
                target=Path(entry[key]).resolve()
                if not target.is_relative_to(project/'data_staging/benchmark_v5') or sha256(target)!=entry[key+'_sha256']: raise ValueError('Automatic staging repair hash/path mismatch')
            automatic[str(Path(entry['original_annotation']).resolve())]=entry
    def absolute(value):
        p=Path(value)
        return (p if p.is_absolute() else project/p).resolve()
    def fixed_copy(value,members):
        if not value: raise ValueError('GEOMETRY_FIXED_COPY requires selected_annotation')
        chosen=absolute(value); staging=project/'data_staging/benchmark_v5'
        if staging.resolve()!=staging or not chosen.is_relative_to(staging): raise ValueError('Fixed copy must be inside data_staging/benchmark_v5')
        provenance=json.loads(chosen.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        source=absolute(provenance['original_annotation'])
        if source not in {absolute(r['source_annotation_path']) for r in members}: raise ValueError('Repair original is not a member of this content group')
        if provenance['original_annotation_sha256']!=sha256(source) or provenance['fixed_annotation_sha256']!=sha256(chosen) or provenance['content_sha256']!=members[0]['content_sha256']:
            raise ValueError('Repair provenance hash mismatch')
        return str(source),str(chosen),provenance
    def choose(original,entry,members):
        decision=entry['decision']; selected=entry['selected']
        if decision=='GEOMETRY_FIXED_COPY': return fixed_copy(selected,members)
        if decision in {'KEEP_A','KEEP_B'}:
            chosen=original['annotation_'+decision[-1].lower()]
            if not chosen: raise ValueError('Requested annotation side does not exist')
        elif decision.startswith('KEEP_PATH:'):
            chosen=decision[len('KEEP_PATH:'):].strip()
            if not chosen: raise ValueError('KEEP_PATH requires a path')
        else: raise ValueError('Source-selection decision required')
        if selected and absolute(selected)!=absolute(chosen): raise ValueError('Selected annotation disagrees with KEEP decision')
        chosen=str(absolute(chosen))
        if absolute(chosen) not in {absolute(r['source_annotation_path']) for r in members}: raise ValueError('Selected raw annotation is not a member of this content group')
        return chosen,chosen,None
    pending=set(); resolved=[]; choices={}; plans=[]; grouped={}
    for row in catalog.values(): grouped.setdefault(row['content_sha256'],[]).append(row)
    for content,items in sorted(grouped.items()):
        members=by_hash[content]
        def answer(r): return supplied.get(r['review_id'],{'decision':'','selected':''})
        exclusions=[r for r in items if answer(r)['decision']=='EXCLUDE']
        if exclusions:
            if any(answer(r)['selected'] for r in exclusions): raise ValueError('EXCLUDE cannot select an annotation')
            choices[content]='EXCLUDE'
            resolved.extend(dict(review_id=r['review_id'],content_sha256=content,decision='EXCLUDE',selected_annotation='') for r in exclusions)
            plans.append(dict(content_sha256=content,status='EXCLUDED',automatic_repairs=[]))
            continue
        conflicts=[r for r in items if r['issue_type']=='ANNOTATION_CONFLICT']
        selected=[]
        for row in conflicts:
            if answer(row)['decision']: selected.append(choose(row,answer(row),members))
            else: pending.add(row['review_id'])
        if conflicts and len(selected)!=len(conflicts):
            plans.append(dict(content_sha256=content,status='AWAITING_SOURCE_SELECTION'))
            continue
        if not conflicts:
            original=items[0]['annotation_a'];selected=[(str(absolute(original)),str(absolute(original)),None)]
        if len({x[:2] for x in selected})!=1: raise ValueError('Inconsistent decisions for content: '+content)
        source,chosen,provenance=selected[0]
        relevant=[r for r in items if r['issue_type']=='GEOMETRY' and absolute(r['annotation_a'])==absolute(source)]
        fixed=[fixed_copy(answer(r)['selected'],members) for r in relevant if answer(r)['decision']=='GEOMETRY_FIXED_COPY']
        if fixed:
            if any(x[0]!=source for x in fixed) or len({x[1] for x in fixed})!=1: raise ValueError('Inconsistent repaired copies')
            source,chosen,provenance=fixed[0]
        automatic_entry=automatic.get(source) if chosen==source else None
        effective=automatic_entry['generated_annotation'] if automatic_entry else chosen
        completed={e['review_id'] for e in automatic_entry['repairs']} if automatic_entry else set()
        geometry_requests=[]
        for row in relevant:
            entry=answer(row);decision=entry['decision'];rid=row['review_id']
            if rid in completed: continue
            if provenance: continue  # The complete replacement must validate below.
            if not decision or decision in {'MANUAL_FIX','AUTO_FIX_APPROVED'}:
                pending.add(rid)
                geometry_requests.append(dict(review_id=rid,decision=decision or 'PENDING',action='AWAIT_MANUAL_COPY' if decision=='MANUAL_FIX' else 'AUTO_FIX_AUTHORIZED_BUT_NOT_EXECUTED' if decision=='AUTO_FIX_APPROVED' else 'REQUIRE_GEOMETRY_DECISION'))
            elif decision=='KEEP_AS_IS':
                geometry_requests.append(dict(review_id=rid,decision=decision,action='RETAIN_WITH_EXPLICIT_EXCEPTION'))
            elif decision in {'KEEP_A','KEEP_B'} or decision.startswith('KEEP_PATH:'):
                other=choose(row,entry,members)
                if other[0]!=source: raise ValueError('Geometry choice contradicts conflict source')
            else: raise ValueError('Unsupported geometry decision')
        record=members[0];w,h=record['width'],record['height']
        data=json.loads(Path(effective).read_text(encoding='utf-8-sig'))
        if data.get('imageWidth')!=w or data.get('imageHeight')!=h: raise ValueError('Selected annotation dimensions mismatch')
        annotation=parse_labelme(Path(effective),Path(items[0]['image_path']),w,h,{0:'tree',1:'TreeGroup'})
        covered={json.loads(r['geometry_issue'])['shape_index']:r for r in relevant if r.get('geometry_issue')}
        exceptions=[]
        for index,obj in enumerate(annotation.objects,1):
            if obj.class_id not in {0,1} or obj.shape_type!='polygon': raise ValueError('Selected annotation has unsupported labels/shapes')
            try:
                valid_polygon(obj.polygon,w,h)
                valid_polygon([(round(x/w,8)*w,round(y/h,8)*h) for x,y in obj.polygon],w,h)
            except ValueError as exc:
                if provenance or index not in covered: raise
                row=covered[index];decision=answer(row)['decision']
                if decision=='KEEP_AS_IS' and str(exc)=='self-intersecting polygon': exceptions.append(row['review_id'])
                else: pending.add(row['review_id'])
        choices[content]=chosen
        for row in items:
            entry=answer(row)
            if entry['decision'] and row['review_id'] not in pending:
                resolved.append(dict(review_id=row['review_id'],content_sha256=content,decision=entry['decision'],selected_annotation=chosen,provenance=provenance))
        plans.append(dict(content_sha256=content,original_annotation=source,selected_annotation=chosen,effective_generated_annotation=effective,automatic_repairs=automatic_entry['repairs'] if automatic_entry else [],geometry_requests=geometry_requests,explicit_geometry_exceptions=exceptions,status='PENDING' if any(r['review_id'] in pending for r in items) else 'REVIEW_RESOLVED'))
    return dict(decision_file=str(Path(path).resolve()),decision_file_sha256=sha256(path),resolved=resolved,pending_review_ids=sorted(pending),selected_by_content=choices,content_plans=plans,build_authorized=False)
