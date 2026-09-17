"""Stage safe repairs and render the remaining human workload; never build v5."""
import csv
import html
import json
from pathlib import Path
from collections import Counter

from PIL import Image
from .reproducible import sha256,write_json
from .review import FIELDS,overlay,page,stats
from .staging_repairs import repair_cases,REPAIR_STATUS

EXTRA=['self_intersection_severity','review_status','linked_geometry_review_ids','geometry_decisions']


def write_rows(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def reduce_review(project):
    project=Path(project).resolve(); root=project/'results/benchmark_v5_review'
    if (project/'config/benchmark_v5_decisions.csv').exists(): raise ValueError('Final review is initialized; preserve Excel edits and use --validate-decisions/--finalize-decisions')
    if root.resolve()!=root: raise ValueError('Unsafe review output')
    catalog=json.loads((root/'review_catalog.json').read_text(encoding='utf-8'))
    records=json.loads((project/'results/benchmark_v5_audit/provenance.json').read_text(encoding='utf-8'))
    details=json.loads((root/'geometry_details.json').read_text(encoding='utf-8'))
    for name in ('review_decisions.csv','human_review_required.csv'):
        path=root/name
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as f:
                if any(any(r.get(k,'') for k in ('human_decision','selected_annotation','notes','geometry_decisions')) for r in csv.DictReader(f)):
                    raise ValueError('Preserve existing human edits before refreshing: '+str(path))
    snapshots={}
    for record in records:
        for key,hashkey in [('source_image_path','content_sha256'),('source_annotation_path','annotation_sha256')]:
            if record[key]:
                snapshots[record[key]]=sha256(record[key])
                if snapshots[record[key]]!=record[hashkey]: raise ValueError('Source changed since audit')
    v4=project/'datasets/benchmark_v4'
    frozen={str(p):sha256(p) for p in v4.rglob('*') if p.is_file()}
    manifest=repair_cases(project,catalog,records)
    repaired={e['review_id']:entry for entry in manifest for e in entry['repairs']}
    conflicts={r['content_sha256']:r for r in catalog if r['issue_type']=='ANNOTATION_CONFLICT'}
    severity={'minor/local':'MINOR','clearly erroneous':'ERRONEOUS','complex/ambiguous':'AMBIGUOUS'}
    intersections=[];overlaps=[]
    for row in catalog:
        row.update(self_intersection_severity='',review_status='UNRESOLVED',linked_geometry_review_ids='',geometry_decisions='')
        if row['issue_type']!='GEOMETRY': continue
        issue=json.loads(row['geometry_issue'])
        row['self_intersection_severity']=severity.get(issue['self_intersection_assessment'],'')
        if row['review_id'] in repaired:
            row['review_status']=REPAIR_STATUS
            row['repair_provenance']=repaired[row['review_id']]
        if row['self_intersection_severity']: intersections.append(row)
        conflict=conflicts.get(row['content_sha256'])
        if conflict:
            clean=[]
            for side in ('a','b'):
                annotation=conflict['annotation_'+side]
                independent=[d for d in details if d['annotation']==annotation]
                same_issue=annotation==row['annotation_a']
                if not same_issue and not independent: clean.append('KEEP_'+side.upper())
            overlaps.append(dict(content_sha256=row['content_sha256'],conflict_review_id=conflict['review_id'],geometry_review_id=row['review_id'],selecting_annotation_removes_geometry_problem=bool(clean),annotation_choices_without_this_problem=';'.join(clean),geometry_status=row['review_status'],resolved_by_safe_staging_repair=row['review_id'] in repaired,meaning='Annotation choices without any flagged geometry before staging; repaired duplicate cases are resolved regardless of source selection'))
    unresolved=[]
    for row in catalog:
        if row['review_status']==REPAIR_STATUS: continue
        if row['issue_type']=='GEOMETRY' and row['content_sha256'] in conflicts: continue
        unresolved.append(dict(row))
    for row in unresolved:
        if row['issue_type']=='ANNOTATION_CONFLICT':
            linked=[g for g in intersections if g['content_sha256']==row['content_sha256']]
            row['linked_geometry_review_ids']=';'.join(g['review_id'] for g in linked)
            if linked: row['geometry_issue']=json.dumps([{'review_id':g['review_id'],'annotation':g['annotation_a'],'geometry_issue':json.loads(g['geometry_issue'])} for g in linked])
            row['self_intersection_severity']=';'.join(sorted({g['self_intersection_severity'] for g in linked}))
    fields=FIELDS[:FIELDS.index('recommended_action')]+['self_intersection_severity']+FIELDS[FIELDS.index('recommended_action'):]+['review_status','linked_geometry_review_ids','geometry_decisions']
    write_rows(root/'review_decisions.csv',catalog,fields)
    write_rows(root/'human_review_required.csv',unresolved,fields)
    overlap_fields=['content_sha256','conflict_review_id','geometry_review_id','selecting_annotation_removes_geometry_problem','annotation_choices_without_this_problem','geometry_status','resolved_by_safe_staging_repair','meaning']
    write_rows(root/'review_overlap_report.csv',overlaps,overlap_fields)
    write_json(root/'review_catalog.json',catalog)
    write_json(root/'safe_duplicate_repairs.json',manifest)
    events=[e|{'source_hash':m['original_annotation_sha256'],'original_annotation':m['original_annotation'],'repaired_staging_hash':m['staged_annotation_sha256'] if e['repair_type']=='RAW_CONSECUTIVE_DUPLICATE' else m['generated_annotation_sha256'],'repaired_staging_path':m['staged_annotation'] if e['repair_type']=='RAW_CONSECUTIVE_DUPLICATE' else m['generated_annotation'],'status':REPAIR_STATUS} for m in manifest for e in m['repairs']]
    write_json(root/'safe_duplicate_repair_events.json',events)
    links=[]
    for row in unresolved:
        identity=row['review_id']; directory=root/'Unresolved'/identity;directory.mkdir(parents=True,exist_ok=True)
        with Image.open(row['image_path']) as im: base=im.convert('RGB')
        a=json.loads(Path(row['annotation_a']).read_text(encoding='utf-8-sig'))
        b=json.loads(Path(row['annotation_b']).read_text(encoding='utf-8-sig')) if row['annotation_b'] else None
        panels=[('original',base),('annotation_A',overlay(base,[(a,'#00e5ff')]))]
        if b: panels.append(('annotation_B',overlay(base,[(b,'#ff4fa3')])))
        panels.append(('combined',overlay(base,[(a,'#00e5ff')]+([(b,'#ff4fa3')] if b else []))))
        related=[d for d in details if d['annotation'] in {row['annotation_a'],row['annotation_b']}]
        for d in related:
            source=a if d['annotation']==row['annotation_a'] else b
            name=('A' if source is a else 'B')+f'_polygon_{d["shape_index"]}'
            im=overlay(base,[(source,'#ff4fa3')],d['shape_index'],d['checks']['raw']['crossings'])
            panels.append((name+'_highlight',im))
            points=source['shapes'][d['shape_index']-1]['points'];xs,ys=zip(*points)
            box=(max(0,int(min(xs))-20),max(0,int(min(ys))-20),min(base.width,int(max(xs))+21),min(base.height,int(max(ys))+21))
            crop=im.crop(box); crop.resize((crop.width*3,crop.height*3)).save(directory/(name+'_zoom.jpg'),quality=96)
        for name,im in panels: im.save(directory/(name+'.jpg'),quality=94)
        metadata={k:row[k] for k in fields};metadata['annotation_A_statistics']=stats(a)
        if b: metadata['annotation_B_statistics']=stats(b)
        body='<a href="../../index.html">All unresolved reviews</a><p>A: cyan; B: magenta. Highlight panels: problem polygon magenta, vertices/crossing edges yellow. Raw source overlays; automatic duplicate repairs do not change the visible boundary. Shape indices are 1-based.</p>'
        body+='<p>Conflict: KEEP_A / KEEP_B / KEEP_PATH / EXCLUDE. Self-intersection: KEEP_AS_IS / MANUAL_FIX / EXCLUDE / AUTO_FIX_APPROVED. No self-intersection repair is executed by this review workflow.</p>'
        body+='<pre>'+html.escape(json.dumps(metadata,indent=2,ensure_ascii=False))+'</pre><div class="grid">'
        body+=''.join(f'<div>{name}<a href="{name}.jpg"><img loading="lazy" src="{name}.jpg"></a></div>' for name,_ in panels)+'</div>'
        for zoom in directory.glob('*_zoom.jpg'): body+=f'<h2>{zoom.stem}</h2><img loading="lazy" src="{zoom.name}">'
        page(directory/'index.html',identity,body)
        links.append((row,f'Unresolved/{identity}/index.html'))
    summary=dict(safe_duplicate_vertex_repairs_applied=len(events),raw_duplicate_repairs=sum(e['repair_type']=='RAW_CONSECUTIVE_DUPLICATE' for e in events),export_rounding_repairs=sum(e['repair_type']=='EXPORT_ROUNDING_CONSECUTIVE_DUPLICATE' for e in events),staged_annotation_files=len(manifest),unresolved_conflict_groups=len(conflicts),unresolved_self_intersection_polygons=len(intersections),self_intersection_severity=dict(Counter(g['self_intersection_severity'] for g in intersections)),overlapping_conflict_geometry_links=len(overlaps),overlapping_content_groups=len({o['content_sha256'] for o in overlaps}),unresolved_conflict_self_intersection_overlaps=sum(g['content_sha256'] in conflicts for g in intersections),total_human_review_items=len(unresolved),benchmark_v5_still_blocked=True)
    body='<p>Only safe duplicate vertices have been repaired, in staging. Raw source and benchmark_v4 are immutable. All self-intersections remain unresolved.</p><p><a href="human_review_required.csv">Reduced human review CSV</a> | <a href="review_overlap_report.csv">Overlap report</a> | <a href="safe_duplicate_repair_events.json">Automatic repair provenance</a> | <a href="README.md">Instructions</a></p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><table><tr><th>ID</th><th>Issue</th><th>Annotators</th><th>Image</th><th>Severity</th></tr>'
    for row,link in links:
        body+='<tr>'+f'<td><a href="{link}">{row["review_id"]}</a></td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [row['issue_type'],row['annotator_a']+' / '+row['annotator_b'],Path(row['image_path']).name,row['self_intersection_severity']])+'</tr>'
    page(root/'index.html','benchmark_v5: unresolved human review',body+'</table>')
    write_json(root/'reduced_workload_summary.json',summary)
    if any(sha256(p)!=h for p,h in snapshots.items()): raise ValueError('Raw source integrity failed')
    if frozen!={str(p):sha256(p) for p in v4.rglob('*') if p.is_file()}: raise ValueError('benchmark_v4 integrity failed')
    write_json(root/'staging_repair_integrity.json',dict(raw_files_verified=len(snapshots),raw_unchanged=True,benchmark_v4_files_verified=len(frozen),benchmark_v4_unchanged=True))
    return summary


if __name__=='__main__': print(json.dumps(reduce_review(Path.cwd()),indent=2))
