import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from src.dataset.reproducible import audit_sources,sha256
from src.dataset.final_decisions import validate_decisions,write_table,CANONICAL_FIELDS,auto_uncross,finalize_decisions
from src.dataset.reviewed_build import build_reviewed,grouped_split


@pytest.fixture
def workflow(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();config=tmp_path/'config';config.mkdir()
    def source(folder,name,color,points):
        image=raw/folder/(name+'.png');image.parent.mkdir(exist_ok=True)
        Image.new('RGB',(16,16),color).save(image)
        image.with_suffix('.json').write_text(json.dumps(dict(imagePath=image.name,imageWidth=16,imageHeight=16,shapes=[dict(label='Tree',shape_type='polygon',points=points)])))
        return image.with_suffix('.json')
    square=[[1,1],[12,1],[12,12],[1,12]]
    a=source('A','frame0_0','red',square)
    b=source('B','frame0_0','red',[[2,2],[11,2],[11,11],[2,11]])
    g=source('A','frame1_0','green',[[1,1],[12,10],[1,12],[10,1]])
    safe=source('A','frame2_0','blue',square[:2]+[square[1]]+square[2:])
    source('A','frame3_0','yellow',square);source('A','frame4_0','purple',square)
    rules=dict(raw_root=str(raw),exclude_labels=['zmladenie'],exclude_shape_types=['circle'])
    (config/'dataset_sources.yaml').write_text(yaml.safe_dump(rules))
    report,records,_=audit_sources(raw,rules)
    by_annotation={r['source_annotation_path']:r for r in records}
    def row(identity,kind,path,status='UNRESOLVED'):
        record=by_annotation[str(path)]
        return dict(review_id=identity,issue_type=kind,content_sha256=record['content_sha256'],annotation_a=str(path),annotation_a_sha256=sha256(path),annotation_b='',annotation_b_sha256='',image_path=record['source_image_path'],geometry_issue=json.dumps({'shape_index':1}),self_intersection_severity='MINOR',review_status=status)
    c=row('C-test','ANNOTATION_CONFLICT',a);c.update(annotation_b=str(b),annotation_b_sha256=sha256(b))
    catalog=[c,row('G-test','GEOMETRY',g),row('G-safe','GEOMETRY',safe,'AUTO_REPAIRED_DUPLICATE_VERTEX')]
    (config/'benchmark_v5_review_catalog.json').write_text(json.dumps(catalog))
    (config/'benchmark_v5_source_snapshot.json').write_text(json.dumps(dict(raw_root=str(raw),files={p.relative_to(raw).as_posix():sha256(p) for p in raw.rglob('*') if p.is_file()})))
    rows=[{k:r.get(k,'') for k in CANONICAL_FIELDS} for r in catalog[:2]]
    def decisions(conflict='',geometry='',selected=''):
        rows[0]['human_decision']=conflict;rows[1]['human_decision']=geometry;rows[1]['selected_annotation']=selected
        write_table(config/'benchmark_v5_decisions.csv',rows,CANONICAL_FIELDS)
    decisions()
    return tmp_path,decisions,g,rows


def test_blank_gate_and_no_output(workflow):
    root,decisions,_,_=workflow
    result=validate_decisions(root)
    assert (result['pending'],result['invalid'],result['complete'])==(2,0,0)
    with pytest.raises(ValueError,match='blocked'): build_reviewed(root,True)
    assert not (root/'data_staging').exists()
    assert not (root/'datasets').exists()


def test_validation_issue_values_missing_duplicate_and_stale(workflow):
    root,decisions,g,rows=workflow
    decisions('KEEP_AS_IS','KEEP_A')
    assert validate_decisions(root)['invalid']==2
    decisions('KEEP_A','AUTO_FIX_APPROVED')
    assert validate_decisions(root)['safe_to_build']
    g.write_text('{}')
    assert validate_decisions(root)['invalid']==2


def test_keep_as_is_build_and_determinism(workflow):
    root,decisions,_,_=workflow;decisions('KEEP_A','KEEP_AS_IS')
    raw_hashes={p:sha256(p) for p in (root/'raw').rglob('*') if p.is_file()}
    v4=root/'datasets/benchmark_v4';v4.mkdir(parents=True);(v4/'sentinel').write_text('unchanged')
    first=build_reviewed(root,True)
    out=root/'datasets/benchmark_v5'
    manifest=(out/'dataset_manifest.csv').read_bytes()
    coco=(out/'coco/annotations/instances_train.json').read_bytes()
    second=build_reviewed(root,True)
    assert first==second and (out/'dataset_manifest.csv').read_bytes()==manifest
    assert (out/'coco/annotations/instances_train.json').read_bytes()==coco
    assert first['decision_file_sha256']==sha256(root/'config/benchmark_v5_decisions.csv')
    assert first['included_image_count']==5
    assert first['explicit_self_intersection_exceptions']
    assert (v4/'sentinel').read_text()=='unchanged'
    assert all(sha256(p)==digest for p,digest in raw_hashes.items())


def test_approved_auto_repair_only_at_build(workflow):
    root,decisions,g,_=workflow;before=g.read_bytes();decisions('KEEP_B','AUTO_FIX_APPROVED')
    assert validate_decisions(root)['safe_to_build']
    assert not (root/'data_staging').exists()
    summary=build_reviewed(root,True)
    assert not summary['explicit_self_intersection_exceptions']
    assert g.read_bytes()==before
    provenance=[json.loads(p.read_text()) for p in (root/'data_staging').rglob('*.provenance.json')]
    assert any(e['repair_type']=='APPROVED_SINGLE_CROSSING_2OPT' for p in provenance for e in p['repairs'])


def test_manual_copy_and_history(workflow):
    root,decisions,g,rows=workflow
    decisions('KEEP_A','MANUAL_FIX')
    assert validate_decisions(root)['invalid']==1
    fixed=root/'data_staging/benchmark_v5/manual.json';fixed.parent.mkdir(parents=True)
    data=json.loads(g.read_text());data['shapes'][0]['points']=[[1,1],[12,1],[12,12],[1,12]];fixed.write_text(json.dumps(data))
    content=rows[1]['content_sha256']
    fixed.with_suffix('.provenance.json').write_text(json.dumps(dict(original_annotation=str(g),original_annotation_sha256=sha256(g),fixed_annotation_sha256=sha256(fixed),content_sha256=content)))
    decisions('KEEP_A','MANUAL_FIX',str(fixed))
    assert validate_decisions(root)['safe_to_build']
    review=root/'human.csv';review.write_bytes((root/'config/benchmark_v5_decisions.csv').read_bytes())
    decisions()
    result=finalize_decisions(root,review)
    assert result['safe_to_build']
    assert len(list((root/'config/benchmark_v5_decision_history').glob('*.csv')))==2


def test_unsupported_repair_and_group_leakage():
    with pytest.raises(ValueError,match='MINOR'): auto_uncross([[0,0],[3,3],[0,3],[3,0]],'AMBIGUOUS')
    records=[dict(content_sha256=str(i),source_image_path=f'/{i%3}/frame{i%3}_{i}.jpg') for i in range(12)]
    assignment,_=grouped_split(records)
    parents={}
    for split,parent in assignment.values():
        assert parent not in parents or parents[parent]==split
        parents[parent]=split


def test_missing_duplicate_unknown_ids_and_excel_notes(workflow):
    root,decisions,g,rows=workflow
    path=root/'config/benchmark_v5_decisions.csv'
    rows[0]['notes']='Czech: p??li?; multi-line\nquoted "note"'
    write_table(path,rows[:1],CANONICAL_FIELDS)
    assert validate_decisions(root)['pending']==2
    write_table(path,rows+[dict(rows[0])],CANONICAL_FIELDS)
    assert validate_decisions(root)['invalid']==1
    extra=dict(rows[0],review_id='UNKNOWN')
    write_table(path,rows+[extra],CANONICAL_FIELDS)
    assert validate_decisions(root)['invalid']==1
    from src.dataset.final_decisions import read_table
    assert read_table(path)[0][0]['notes']==rows[0]['notes']


def test_cli_validation_exit_status_and_no_decisions(workflow,monkeypatch):
    root,decisions,g,rows=workflow
    import src.config
    import main
    monkeypatch.setattr(src.config,'load_config',lambda path:{'_project_root':str(root)})
    assert main.main(['prepare','--dataset','benchmark_v5','--validate-decisions'])==2
    from src.dataset.final_decisions import read_table
    saved,_=read_table(root/'config/benchmark_v5_decisions.csv')
    assert all(not r['human_decision'] and r['decision_status']=='PENDING' for r in saved)
    assert not (root/'datasets').exists()


def test_strict_policy_excludes_instead_of_repairing(workflow):
    root,decisions,g,rows=workflow
    config=root/'config/dataset_sources.yaml'
    rules=yaml.safe_load(config.read_text());rules['benchmark_v5_strict_geometry']=True;config.write_text(yaml.safe_dump(rules))
    decisions('KEEP_A','AUTO_FIX_APPROVED')
    assert validate_decisions(root)['invalid']==1
    decisions('KEEP_A','EXCLUDE')
    assert validate_decisions(root)['safe_to_build']
    summary=build_reviewed(root,True)
    assert summary['included_image_count']==4
    assert summary['explicit_self_intersection_exceptions']=={}
    ids=[]
    for split in ('train','val','test'):
        data=json.loads((root/f'datasets/benchmark_v5/coco/annotations/instances_{split}.json').read_text())
        ids.extend(i['id'] for i in data['images'])
        assert all(i['canonical_image_id']!=rows[1]['content_sha256'] for i in data['images'])
    assert len(set(ids))==len(ids)
