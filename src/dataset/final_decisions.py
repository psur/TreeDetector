"""Excel decision tables, strict validation, and deterministic annotation plans."""
from __future__ import annotations
import copy
import csv
import io
import json
from collections import Counter
from pathlib import Path

from .reproducible import sha256, valid_polygon, audit_sources, write_json
from .staging_repairs import remove_consecutive, generated_points
from .review import geometry

CANONICAL_FIELDS=['review_id','issue_type','content_sha256','human_decision','selected_annotation','notes','decision_status']
ALLOWED={'ANNOTATION_CONFLICT':{'KEEP_A','KEEP_B','EXCLUDE'},'GEOMETRY':{'KEEP_AS_IS','AUTO_FIX_APPROVED','MANUAL_FIX','EXCLUDE'}}


def read_table(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    header=text.splitlines()[0] if text else ''
    delimiter=';' if header.count(';')>header.count(',') else ','
    reader=csv.DictReader(io.StringIO(text),delimiter=delimiter,strict=True)
    if not set(CANONICAL_FIELDS[:6]).issubset(reader.fieldnames or []): raise ValueError('Missing required decision columns')
    try: rows=list(reader)
    except csv.Error as exc: raise ValueError('Malformed CSV: '+str(exc)) from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows): raise ValueError('Malformed CSV row: column count mismatch')
    return rows,reader.fieldnames


def write_table(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter=';',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def auto_uncross(points,severity):
    """Supported: MINOR single proper crossing, resolved by one 2-opt reversal."""
    if severity!='MINOR': raise ValueError('AUTO_FIX_APPROVED supports MINOR single proper crossings only')
    p=[list(v) for v in points]; events=geometry(p)['crossings']
    if len(events)!=1: raise ValueError('AUTO_FIX_APPROVED requires exactly one crossing')
    i,j=[k-1 for k in events[0]['edges']]
    a,b=p[i],p[(i+1)%len(p)];c,d=p[j],p[(j+1)%len(p)]
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0): raise ValueError('Touching/overlapping edges require MANUAL_FIX or KEEP_AS_IS')
    result=p[:i+1]+list(reversed(p[i+1:j+1]))+p[j+1:]
    if geometry(result)['invalid'] or geometry(result)['category']: raise ValueError('Single-crossing repair did not produce a simple polygon')
    return result,dict(repair_type='APPROVED_SINGLE_CROSSING_2OPT',reversed_vertex_indices=list(range(i+1,j+1)),index_base=0,original_vertex_count=len(p),repaired_vertex_count=len(result))


def annotation_plan(project,source,record,geometry_rows,decisions,safe_rows):
    original=json.loads(Path(source).read_text(encoding='utf-8-sig')); data=copy.deepcopy(original)
    provenance=[]; manual=[]
    for row in geometry_rows:
        answer=decisions[row['review_id']]
        if answer['human_decision']=='MANUAL_FIX':
            path=Path(answer['selected_annotation'])
            if not str(answer['selected_annotation']).strip(): raise ValueError('MANUAL_FIX requires selected_annotation')
            path=(path if path.is_absolute() else project/path).resolve()
            staging=project/'data_staging/benchmark_v5'
            if staging.resolve()!=staging or not path.is_relative_to(staging): raise ValueError('MANUAL_FIX must use a staged annotation')
            sidecar=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
            if Path(sidecar['original_annotation']).resolve()!=Path(source).resolve() or sidecar['original_annotation_sha256']!=sha256(source) or sidecar['fixed_annotation_sha256']!=sha256(path) or sidecar['content_sha256']!=record['content_sha256']:
                raise ValueError('MANUAL_FIX provenance does not match source/copy')
            manual.append((path,sidecar))
    if len({str(p) for p,_ in manual})>1: raise ValueError('MANUAL_FIX rows for the same source must select the same full annotation copy')
    if manual:
        data=json.loads(manual[0][0].read_text(encoding='utf-8-sig'));provenance.append(dict(repair_type='HUMAN_MANUAL_COPY',path=str(manual[0][0]),provenance=manual[0][1]))
    w,h=record['width'],record['height']
    if data.get('imageWidth')!=w or data.get('imageHeight')!=h: raise ValueError('Annotation dimensions do not match source image')
    safe_indices={json.loads(r['geometry_issue'])['shape_index'] for r in safe_rows}
    if not manual:
        for index in sorted(safe_indices):
            before=data['shapes'][index-1]['points'];after,removed=remove_consecutive(before)
            if removed:
                data['shapes'][index-1]['points']=after
                provenance.append(dict(repair_type='RAW_CONSECUTIVE_DUPLICATE',shape_index=index,original_vertex_count=len(before),repaired_vertex_count=len(after),removed_vertex_indices=removed,index_base=0))
        for row in geometry_rows:
            if decisions[row['review_id']]['human_decision']=='AUTO_FIX_APPROVED':
                index=json.loads(row['geometry_issue'])['shape_index']
                fixed,event=auto_uncross(data['shapes'][index-1]['points'],row['self_intersection_severity'])
                data['shapes'][index-1]['points']=fixed;provenance.append(event|{'shape_index':index,'review_id':row['review_id']})
    exceptions={json.loads(r['geometry_issue'])['shape_index'] for r in geometry_rows if decisions[r['review_id']]['human_decision']=='KEEP_AS_IS'} if not manual else set()
    for index,shape in enumerate(data.get('shapes',[]),1):
        if shape.get('shape_type','polygon').casefold()!='polygon' or shape.get('label','').strip().casefold() not in {'tree','treegroup'}: raise ValueError('Unsupported annotation label or shape')
        shape['points']=generated_points(shape['points'],w,h)
        if index in safe_indices and not manual:
            before=shape['points'];after,removed=remove_consecutive(before);shape['points']=after
            if removed: provenance.append(dict(repair_type='EXPORT_ROUNDING_CONSECUTIVE_DUPLICATE',shape_index=index,original_vertex_count=len(before),repaired_vertex_count=len(after),removed_vertex_indices=removed,index_base=0))
        try: valid_polygon(shape['points'],w,h)
        except ValueError as exc:
            if str(exc)!='self-intersecting polygon' or index not in exceptions: raise
    return dict(data=data,provenance=provenance,allowed_self_intersections=sorted(exceptions),original_annotation=source,original_annotation_sha256=sha256(source))


def validate_decisions(project,decision_path=None):
    import yaml
    project=Path(project).resolve();config=project/'config'
    path=Path(decision_path) if decision_path else config/'benchmark_v5_decisions.csv'
    if not path.is_absolute(): path=project/path
    catalog=json.loads((config/'benchmark_v5_review_catalog.json').read_text(encoding='utf-8'))
    required={r['review_id']:r for r in catalog if r.get('review_status')!='AUTO_REPAIRED_DUPLICATE_VERTEX'}
    snapshot=json.loads((config/'benchmark_v5_source_snapshot.json').read_text(encoding='utf-8'))
    rules=yaml.safe_load((config/'dataset_sources.yaml').read_text(encoding='utf-8'));raw=Path(rules['raw_root']).resolve()
    problems=[]; changed=[]
    from .scanner import EXTS
    current={p.relative_to(raw).as_posix():p for p in raw.rglob('*') if p.is_file() and p.suffix.casefold() in EXTS|{'.json'}}
    for name,digest in snapshot['files'].items():
        if name not in current or sha256(current[name])!=digest: changed.append(name)
    changed+=sorted(set(current)-set(snapshot['files']))
    if str(raw)!=snapshot['raw_root']: changed.append('RAW_ROOT_CHANGED')
    if changed: problems.append('Source snapshot mismatch: '+', '.join(changed))
    report,records,_=audit_sources(raw,rules)
    try: rows,fields=read_table(path)
    except (ValueError,OSError) as exc: rows=[];problems.append(str(exc))
    by_id={};duplicates=set();unknown=[]
    for row in rows:
        identity=row.get('review_id','').strip()
        if identity in by_id: duplicates.add(identity)
        if identity not in required: unknown.append(identity)
        row['human_decision']=(row.get('human_decision') or '').strip()
        row['selected_annotation']=(row.get('selected_annotation') or '').strip()
        by_id[identity]=row
    results={};plans={};by_content={}
    for record in records: by_content.setdefault(record['content_sha256'],[]).append(record)
    for identity,item in required.items():
        status='PENDING';errors=[];row=by_id.get(identity)
        if row is None: errors.append('Required review ID missing')
        else:
            if row.get('content_sha256')!=item['content_sha256'] or row.get('issue_type')!=item['issue_type']: errors.append('Identity columns do not match catalog')
            if identity in duplicates: errors.append('Duplicate review ID')
            decision=row['human_decision']
            if decision:
                if decision not in ALLOWED[item['issue_type']]: errors.append('Decision not allowed for issue type')
                else: status='COMPLETE'
            if row['selected_annotation'] and decision not in {'KEEP_A','KEEP_B','MANUAL_FIX'}: errors.append('selected_annotation is not used by this decision')
            if rules.get('benchmark_v5_strict_geometry',False) and item['issue_type']=='GEOMETRY' and decision and decision!='EXCLUDE': errors.append('Strict geometry policy requires EXCLUDE for remaining invalid geometry')
        if row is not None and status=='COMPLETE' and not errors and not problems:
            try:
                decision=row['human_decision']
                if decision in {'KEEP_A','KEEP_B'}:
                    source=item['annotation_'+decision[-1].lower()]
                    if not source or source not in {r['source_annotation_path'] for r in by_content.get(item['content_sha256'],[])}: raise ValueError('Invalid conflict alternative')
                    selected=Path(row['selected_annotation']) if row['selected_annotation'] else None
                    if selected and (selected if selected.is_absolute() else project/selected).resolve()!=Path(source).resolve(): raise ValueError('selected_annotation disagrees with KEEP choice')
                if decision=='MANUAL_FIX':
                    source=item['annotation_a'];record=next(r for r in by_content[item['content_sha256']] if r['source_annotation_path']==source)
                    annotation_plan(project,source,record,[item],by_id,[])
                if decision=='AUTO_FIX_APPROVED':
                    source=json.loads(Path(item['annotation_a']).read_text(encoding='utf-8-sig'))
                    index=json.loads(item['geometry_issue'])['shape_index']
                    fixed,_=auto_uncross(source['shapes'][index-1]['points'],item['self_intersection_severity'])
                    valid_polygon(generated_points(fixed,source['imageWidth'],source['imageHeight']),source['imageWidth'],source['imageHeight'])
            except (ValueError,OSError,KeyError,IndexError,TypeError,StopIteration) as exc: errors.append(str(exc))
        for side in ('a','b'):
            source=item.get('annotation_'+side,'')
            if source:
                match=next((r for r in records if r['source_annotation_path']==source),None)
                if match is None or match['annotation_sha256']!=item['annotation_'+side+'_sha256']: errors.append('Catalog source annotation hash mismatch')
        if problems: errors+=problems
        if errors: status='INVALID' if row is not None or problems else 'PENDING'
        results[identity]=dict(review_id=identity,issue_type=item['issue_type'],decision_status=status,errors=errors)
    # Exclusions dominate work, but every required row still needs an explicit valid decision.
    content_items={}
    for item in required.values(): content_items.setdefault(item['content_sha256'],[]).append(item)
    for content,items in content_items.items():
        if any(results[r['review_id']]['decision_status']!='COMPLETE' for r in items): continue
        if any(by_id[r['review_id']]['human_decision']=='EXCLUDE' for r in items):
            plans[content]={'exclude':True};continue
        try:
            conflicts=[r for r in items if r['issue_type']=='ANNOTATION_CONFLICT']
            if conflicts:
                choices=[]
                for row in conflicts:
                    answer=by_id[row['review_id']];side=answer['human_decision'][-1].lower();source=row['annotation_'+side]
                    if not source or source not in {r['source_annotation_path'] for r in by_content[content]}: raise ValueError('KEEP choice does not name a current conflict alternative')
                    if answer['selected_annotation'] and Path(answer['selected_annotation']).resolve()!=Path(source).resolve(): raise ValueError('selected_annotation disagrees with KEEP choice')
                    choices.append(source)
                if len(set(choices))!=1: raise ValueError('Conflicting source selections')
                source=choices[0]
            else: source=items[0]['annotation_a']
            record=next(r for r in by_content[content] if r['source_annotation_path']==source)
            geometry_rows=[r for r in items if r['issue_type']=='GEOMETRY' and r['annotation_a']==source]
            safe=[r for r in catalog if r['annotation_a']==source and r.get('review_status')=='AUTO_REPAIRED_DUPLICATE_VERTEX']
            plans[content]=annotation_plan(project,source,record,geometry_rows,by_id,safe)|{'record':record}
        except (ValueError,OSError,KeyError,IndexError,TypeError) as exc:
            for row in items: results[row['review_id']].update(decision_status='INVALID',errors=[str(exc)])
    for identity in unknown: results['UNKNOWN:'+identity]=dict(review_id=identity,issue_type='UNKNOWN',decision_status='INVALID',errors=['Unknown review ID'])
    counts=Counter(r['decision_status'] for r in results.values())
    safe=counts['PENDING']==counts['INVALID']==0 and not problems
    return dict(decision_file=str(path.resolve()),decision_file_sha256=sha256(path) if path.exists() else '',total_review_items=len(required),complete=counts['COMPLETE'],pending=counts['PENDING'],invalid=counts['INVALID'],conflicts_complete=sum(r['decision_status']=='COMPLETE' and r['issue_type']=='ANNOTATION_CONFLICT' for r in results.values()),geometry_complete=sum(r['decision_status']=='COMPLETE' and r['issue_type']=='GEOMETRY' for r in results.values()),safe_to_build=safe,items=list(results.values()),source_hashes_match=not changed,source_changes=changed,problems=problems,_plans=plans,_records=records,_catalog=catalog,_audit=report)


def public_report(result): return {k:v for k,v in result.items() if not k.startswith('_')}


def print_report(result):
    for key in ['decision_file','total_review_items','complete','pending','invalid']:
        print(key.upper().replace('_',' ')+': '+str(result[key]))
    print('CONFLICTS COMPLETE: '+str(result['conflicts_complete'])+'/'+str(sum(r['issue_type']=='ANNOTATION_CONFLICT' for r in result['items'])))
    print('GEOMETRY COMPLETE: '+str(result['geometry_complete'])+'/'+str(sum(r['issue_type']=='GEOMETRY' for r in result['items'])))
    print('SAFE TO BUILD BENCHMARK_V5: '+('YES' if result['safe_to_build'] else 'NO'))
    if not result['safe_to_build']:
        for row in result['items']:
            if row['decision_status']!='COMPLETE': print(row['review_id'])


def validation_command(project,decision_path=None):
    result=validate_decisions(project,decision_path);print_report(result)
    root=Path(project)/'results/benchmark_v5_review'
    write_json(root/'decision_validation.json',public_report(result))
    (root/'unresolved_review_ids.txt').write_text('\n'.join(r['review_id'] for r in result['items'] if r['decision_status']!='COMPLETE')+'\n',encoding='utf-8')
    # Statuses are computed metadata. Preserve every human value exactly.
    path=Path(result['decision_file'])
    if path.exists():
        rows,fields=read_table(path);statuses={r['review_id']:r['decision_status'] for r in result['items']}
        for row in rows: row['decision_status']=statuses.get(row['review_id'],'INVALID')
        if 'decision_status' not in fields: fields.append('decision_status')
        write_table(path,rows,fields)
        result['decision_file_sha256']=sha256(path)
        write_json(root/'decision_validation.json',public_report(result))
    import re
    index=root/'index.html'
    if index.exists():
        text=index.read_text(encoding='utf-8')
        for item in result['items']:
            identity=item['review_id'];start=text.find('<tr><td><a href="Unresolved/'+identity)
            if start>=0:
                end=text.find('</tr>',start)
                segment=re.sub(r'<td>(PENDING|COMPLETE|INVALID)</td>', '<td>'+item['decision_status']+'</td>',text[start:end])
                text=text[:start]+segment+text[end:]
            page=root/'Unresolved'/identity/'index.html'
            if page.exists():
                contents=page.read_text(encoding='utf-8')
                contents=re.sub(r'Status: (PENDING|COMPLETE|INVALID)', 'Status: '+item['decision_status'],contents)
                page.write_text(contents,encoding='utf-8')
        index.write_text(text,encoding='utf-8')
    return result


def finalize_decisions(project,source):
    """Explicitly import completed human edits; preserve the previous canonical CSV."""
    import shutil
    project=Path(project).resolve()
    if not source: raise ValueError('--finalize-decisions requires --decisions <review CSV>')
    source=Path(source);source=source if source.is_absolute() else project/source
    result=validation_command(project,source)
    if not result['safe_to_build']: return result
    canonical=project/'config/benchmark_v5_decisions.csv'
    history=project/'config/benchmark_v5_decision_history';history.mkdir(exist_ok=True)
    if canonical.exists(): shutil.copy2(canonical,history/(sha256(canonical)+'.csv'))
    rows,_=read_table(source)
    write_table(canonical,sorted(rows,key=lambda r:r['review_id']),CANONICAL_FIELDS)
    shutil.copy2(canonical,history/(sha256(canonical)+'.csv'))
    return validation_command(project)
