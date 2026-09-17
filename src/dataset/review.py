"""Offline human review assets. Never writes source annotations or builds datasets."""
from __future__ import annotations
import argparse
import csv
import html
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw
from .reproducible import sha256, write_json

FIELDS = 'review_id content_sha256 issue_type annotator_a annotator_b image_path annotation_a annotation_b tree_count_a treegroup_count_a tree_count_b treegroup_count_b geometry_issue recommended_action human_decision selected_annotation notes'.split()


def area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2


def geometry(points):
    p = [tuple(map(float, x)) for x in points]
    if any(len(x)!=2 or not all(math.isfinite(v) for v in x) for x in p):
        return dict(category='D', invalid=True, crossings=[], consecutive=0, closing=False)
    closing = len(p)>1 and p[0]==p[-1]
    ring = p[:-1] if closing else p[:]
    clean=[]
    for point in ring:
        if not clean or point != clean[-1]: clean.append(point)
    if len(clean)>1 and clean[0]==clean[-1]: clean.pop()
    consecutive=len(ring)-len(clean)
    elsewhere=len(set(clean))<len(clean)
    repeated=len(set(p))<len(p)
    category = ('C' if elsewhere else 'B' if consecutive else 'A' if closing else 'D') if repeated else ''
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c): return cross(a,b,c)==0 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
    crossings=[]
    edges=list(zip(clean,clean[1:]+clean[:1]))
    for i,(a,b) in enumerate(edges):
        for j in range(i+1,len(edges)):
            c,d=edges[j]
            if j==i+1 or (i==0 and j==len(edges)-1):
                # Adjacent edges may share an endpoint, but may not backtrack.
                if j==i+1 and a!=c and on(a,b,d) and d!=b:
                    crossings.append(dict(edges=[i+1,j+1],points=[a,b,c,d],kind='backtrack'))
                continue
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                crossings.append(dict(edges=[i+1,j+1],points=[a,b,c,d],kind='crossing_or_touch'))
    invalid = len(set(clean))<3 or area(clean)==0 or elsewhere or bool(crossings)
    return dict(category=category,invalid=invalid,crossings=crossings,consecutive=consecutive,closing=closing,
                removal_preserves_exact_boundary=bool(consecutive),valid_after_duplicate_removal=not invalid)


def stats(data):
    polygons=[s for s in data['shapes'] if s.get('shape_type','polygon').casefold()=='polygon']
    counts=Counter(s.get('label','').strip().casefold() for s in polygons)
    return dict(tree_count=counts['tree'],treegroup_count=counts['treegroup'],polygon_count=len(polygons),
                total_shape_count=len(data['shapes']),summed_absolute_shoelace_area_px2=sum(area(s['points']) for s in polygons))


def overlay(base, sets, highlight=None, crossings=()):
    im=base.copy(); draw=ImageDraw.Draw(im)
    for data,color in sets:
        for i,s in enumerate(data['shapes'],1):
            pts=[tuple(p) for p in s.get('points',[])]
            if len(pts)<2: continue
            kind=s.get('shape_type','polygon')
            c=color if highlight is None or i==highlight else '#888888'
            if kind=='circle':
                x,y=pts[0]; r=math.dist(pts[0],pts[1]); draw.ellipse((x-r,y-r,x+r,y+r),outline=c,width=3)
            else: draw.line(pts+pts[:1],fill=c,width=4 if i==highlight else 2)
            draw.text(pts[0],f'{i}:{s.get("label", "")}',fill=c,stroke_width=1,stroke_fill='black')
            if i==highlight:
                for x,y in pts: draw.ellipse((x-3,y-3,x+3,y+3),outline='#ffff00',width=2)
    for event in crossings:
        a,b,c,d=event['points']
        for x,y in ((a,b),(c,d)): draw.line([x,y],fill='#ffff00',width=6)
    return im


def page(path,title,body):
    path.write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui;margin:24px;background:#17191d;color:#eee}a{color:#8dccff}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:8px;border:1px solid #777}table{border-collapse:collapse}</style><h1>'+html.escape(title)+'</h1>'+body,encoding='utf-8')


def build_review(project):
    project=Path(project).resolve(); audit_dir=project/'results/benchmark_v5_audit'
    audit=json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    records=json.loads((audit_dir/'provenance.json').read_text(encoding='utf-8'))
    out=project/'results/benchmark_v5_review'
    if out.exists(): raise ValueError('Review package already exists; preserve human decisions and choose a fresh destination in code')
    if out.resolve()!=out or not out.is_relative_to(project): raise ValueError('Unsafe review output')
    data={}; snapshots={}
    for r in records:
        for key,hashkey in [('source_image_path','content_sha256'),('source_annotation_path','annotation_sha256')]:
            p=r[key]
            if not p: continue
            digest=sha256(p)
            if digest!=r[hashkey]: raise ValueError('Source changed since audit: '+p)
            snapshots[p]=digest
        if r['source_annotation_path']:
            data[r['source_annotation_path']]=json.loads(Path(r['source_annotation_path']).read_text(encoding='utf-8-sig'))
    out.mkdir(parents=True)
    rows=[]; catalog=[]; sections={x:[] for x in ['Jakub_vs_Marek','Oliver_internal','Other','Geometry']}
    group_counts=Counter(); issues=[]
    def row(identity,digest,kind,a,b=None):
        result=dict.fromkeys(FIELDS,''); result.update(review_id=identity,content_sha256=digest,issue_type=kind,image_path=a['source_image_path'],annotation_a=a['source_annotation_path'],annotator_a=a.get('annotator',a.get('source_annotator','')))
        for side,s in [('a',a),('b',b)]:
            if s:
                result['annotation_'+side]=s['source_annotation_path']; result['annotator_'+side]=s.get('annotator',s.get('source_annotator',''))
                counts=stats(data[s['source_annotation_path']])
                for k in ['tree_count','treegroup_count']: result[k+'_'+side]=counts[k]
        return result
    for group in audit['annotation_conflicts']:
        names={s['annotator'].casefold() for s in group['sources']}
        section='Jakub_vs_Marek' if names=={'jakub','marek'} else 'Oliver_internal' if names=={'oliver'} else 'Other'
        group_counts[section]+=1
        for pair,(a,b) in enumerate(combinations(group['sources'],2),1):
            identity='C-'+group['content_sha256'][:16]+f'-{pair}'
            directory=out/section/identity; directory.mkdir(parents=True)
            with Image.open(a['source_image_path']) as im: base=im.convert('RGB')
            ad,bd=data[a['source_annotation_path']],data[b['source_annotation_path']]
            panels=[('original',base),('annotation_A',overlay(base,[(ad,'#00e5ff')])),('annotation_B',overlay(base,[(bd,'#ff4fa3')])),('combined',overlay(base,[(ad,'#00e5ff'),(bd,'#ff4fa3')]))]
            for name,im in panels: im.save(directory/(name+'.jpg'),quality=94)
            metadata=dict(content_sha256=group['content_sha256'],A=a|stats(ad),B=b|stats(bd))
            page(directory/'index.html',identity,'<a href="../../index.html">All reviews</a><p>A: cyan. B: magenta. Polygon numbers are 1-based JSON shape indices. Areas are summed absolute shoelace areas, not union areas; invalid rings may cancel.</p><pre>'+html.escape(json.dumps(metadata,indent=2,ensure_ascii=False))+'</pre><div class="grid">'+''.join(f'<div>{name}<a href="{name}.jpg"><img loading="lazy" src="{name}.jpg"></a></div>' for name,_ in panels)+'</div>')
            r=row(identity,group['content_sha256'],'ANNOTATION_CONFLICT',a,b); r['recommended_action']='REVIEW_BOTH'; rows.append(r)
            catalog.append(r|{'annotation_a_sha256':snapshots[r['annotation_a']],'annotation_b_sha256':snapshots[r['annotation_b']]})
            sections[section].append((identity,f'{section}/{identity}/index.html'))
    for record in records:
        annotation=record['source_annotation_path']
        if not annotation: continue
        shapes=data[annotation]['shapes']; image_issues=[]
        w,h=record['width'],record['height']
        for index,s in enumerate(shapes,1):
            if s.get('shape_type','polygon').casefold()!='polygon': continue
            raw=[tuple(p) for p in s['points']]
            normalized=[(min(max(x,0),w),min(max(y,0),h)) if -1<=x<=w+1 and -1<=y<=h+1 else (x,y) for x,y in raw]
            serialized=[(round(x/w,8)*w,round(y/h,8)*h) for x,y in normalized]
            checks={name:geometry(p) for name,p in [('raw',raw),('normalized',normalized),('serialized',serialized)]}
            if not any(g['category'] or g['invalid'] for g in checks.values()): continue
            issue=dict(annotation=annotation,image=record['source_image_path'],shape_index=index,label=s['label'],prior_exclusion=record['exclusion_reason'],checks=checks)
            issue['self_intersection_assessment']='complex/ambiguous' if any(g['crossings'] for g in checks.values()) else ''
            issue['assessment_basis']='Conservative technical triage; human interpretation required. No repair inferred.'
            image_issues.append(issue); issues.append(issue)
        if not image_issues: continue
        with Image.open(record['source_image_path']) as im: base=im.convert('RGB')
        for issue in image_issues:
            index=issue['shape_index']; identity='G-'+snapshots[annotation][:16]+f'-{index}'
            directory=out/'Geometry'/identity; directory.mkdir(parents=True)
            base.save(directory/'original.jpg',quality=94)
            overlay(base,[(data[annotation],'#ff4fa3')],index,issue['checks']['raw']['crossings']).save(directory/'overlay.jpg',quality=94)
            # Full-resolution crop keeps small local defects visible.
            pts=shapes[index-1]['points']; xs,ys=zip(*pts)
            box=(max(0,int(min(xs))-20),max(0,int(min(ys))-20),min(w,int(max(xs))+21),min(h,int(max(ys))+21))
            with Image.open(directory/'overlay.jpg') as im: im.crop(box).save(directory/'detail.jpg',quality=96)
            page(directory/'index.html',identity,'<a href="../../index.html">All reviews</a><p>Problem polygon: magenta; vertices and intersecting edges: yellow; other polygons: gray. Overlay uses unmodified raw coordinates. Compare normalized/export checks below.</p><pre>'+html.escape(json.dumps(issue,indent=2,ensure_ascii=False))+'</pre><div class="grid"><img src="original.jpg"><img src="overlay.jpg"></div><h2>Polygon detail</h2><img src="detail.jpg">')
            r=row(identity,record['content_sha256'],'GEOMETRY',record)
            r['geometry_issue']=json.dumps({'shape_index':index,'checks':issue['checks'],'self_intersection_assessment':issue['self_intersection_assessment']},separators=(',',':'))
            g=issue['checks']['raw']; all_valid=not any(x['invalid'] for x in issue['checks'].values())
            r['recommended_action']='LIKELY_VALID_CLOSING_VERTEX' if g['category']=='A' and all_valid else 'SAFE_REMOVE_DUPLICATE_VERTEX' if any(x['consecutive'] for x in issue['checks'].values()) and all_valid else 'MANUAL_GEOMETRY_REPAIR'
            rows.append(r); catalog.append(r|{'annotation_a_sha256':snapshots[annotation],'annotation_b_sha256':''}); sections['Geometry'].append((identity,f'Geometry/{identity}/index.html'))
    repeated=[i for i in issues if i['checks']['raw']['category']]
    summary=dict(conflict_groups={s:group_counts[s] for s in sections if s!='Geometry'},geometry_polygons=len(issues),
                 repeated_vertex_raw=dict(closing_vertex_only=sum(i['checks']['raw']['category']=='A' for i in repeated),consecutive_duplicate=sum(i['checks']['raw']['category']=='B' for i in repeated),other=sum(i['checks']['raw']['category'] in {'C','D'} for i in repeated),genuinely_invalid=sum(i['checks']['raw']['invalid'] for i in repeated)),
                 prior_repeated_vertex_images=sum(r['exclusion_reason']=='repeated polygon vertex' for r in records),
                 self_intersections=dict(total=sum(bool(i['self_intersection_assessment']) for i in issues),minor_local=0,clearly_erroneous=0,ambiguous=sum(bool(i['self_intersection_assessment']) for i in issues)),
                 recommendations=dict(Counter(r['recommended_action'] for r in rows)),review_rows=len(rows),benchmark_v5_still_blocked=True)
    with (out/'review_decisions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    write_json(out/'review_catalog.json',catalog); write_json(out/'geometry_details.json',issues); write_json(out/'summary.json',summary)
    write_json(out/'source_integrity.json',dict(before=snapshots,unchanged_after=all(sha256(p)==h for p,h in snapshots.items())))
    body='<p>No decisions or repairs have been applied. Edit a copy of review_decisions.csv; leave recommendations separate from decisions. All images are local. Self-intersection severity is conservatively ambiguous pending human review.</p><p><a href="review_decisions.csv">Review CSV</a> | <a href="summary.json">Summary</a> | <a href="geometry_details.json">Geometry details</a></p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'
    for section,links in sections.items():
        (out/section).mkdir(exist_ok=True)
        body+=f'<h2>{section} ({len(links)} reviews)</h2><ul>'+''.join(f'<li><a href="{link}">{name}</a></li>' for name,link in links)+'</ul>'
    page(out/'index.html','benchmark_v5 human review',body)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',default='.'); args=parser.parse_args()
    print(json.dumps(build_review(args.project),indent=2))
