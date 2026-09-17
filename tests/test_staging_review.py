import copy
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset.review import FIELDS
from src.dataset.staging_repairs import remove_consecutive, generated_points, repair_cases
from src.dataset.review_decisions import read_decisions
from src.dataset.reproducible import sha256


def test_remove_only_exact_consecutive_duplicates():
    original=[[0,0],[10,0],[10,0],[10,10],[0,10],[0,0]]
    fixed,removed=remove_consecutive(original)
    assert removed==[2]
    assert fixed==[[0,0],[10,0],[10,10],[0,10],[0,0]]
    assert original[1:3]==[[10,0],[10,0]]
    nonconsecutive=[[0,0],[10,0],[10,10],[10,0],[0,10]]
    assert remove_consecutive(nonconsecutive)==(nonconsecutive,[])
    with pytest.raises(ValueError,match='independent issue'):
        remove_consecutive([[0,0],[10,10],[10,10],[0,10],[10,0]])


def test_raw_and_export_repairs_provenance_and_idempotence(tmp_path):
    source=tmp_path/'source.json'
    shapes=[dict(label='Tree',shape_type='polygon',points=[[1,1],[12,1],[12,1],[12,12],[1,12]]),dict(label='Tree',shape_type='polygon',points=[[2,2],[10,2],[10.000000001,2],[10,10],[2,10]])]
    data=dict(imageWidth=16,imageHeight=16,imagePath='source.png',shapes=shapes)
    source.write_text(json.dumps(data)); before=source.read_bytes();digest=sha256(source)
    rows=[dict(issue_type='GEOMETRY',recommended_action='SAFE_REMOVE_DUPLICATE_VERTEX',annotation_a=str(source),annotation_a_sha256=digest,review_id=f'G-{i}',geometry_issue=json.dumps({'shape_index':i})) for i in (1,2)]
    records=[dict(source_annotation_path=str(source),annotation_sha256=digest,source_image_path='source.png',content_sha256='content',width=16,height=16)]
    manifest=repair_cases(tmp_path,rows,records); entry=manifest[0]
    assert source.read_bytes()==before
    raw=json.loads(Path(entry['staged_annotation']).read_text()); exported=json.loads(Path(entry['generated_annotation']).read_text())
    assert len(raw['shapes'][0]['points'])==4
    assert raw['shapes'][1]==data['shapes'][1]  # distinct raw vertex is retained
    assert len(exported['shapes'][1]['points'])==4
    assert {e['repair_type'] for e in entry['repairs']}=={'RAW_CONSECUTIVE_DUPLICATE','EXPORT_ROUNDING_CONSECUTIVE_DUPLICATE'}
    assert all(e['removed_vertex_indices']==[2] for e in entry['repairs'])
    assert repair_cases(tmp_path,rows,records)==manifest
    Path(entry['staged_annotation']).write_text('{}')
    with pytest.raises(ValueError,match='overwrite changed staging'): repair_cases(tmp_path,rows,records)


@pytest.fixture
def overlapping_case(tmp_path):
    image=tmp_path/'source.png';Image.new('RGB',(16,16)).save(image)
    a=tmp_path/'a.json';b=tmp_path/'b.json'
    for p,points in [(a,[[1,1],[12,10],[1,12],[10,1]]),(b,[[1,1],[12,1],[12,12],[1,12]])]:
        p.write_text(json.dumps(dict(imagePath=image.name,imageWidth=16,imageHeight=16,shapes=[dict(label='Tree',shape_type='polygon',points=points)])))
    digest=sha256(image)
    conflict=dict.fromkeys(FIELDS,'');conflict.update(review_id='C-1',issue_type='ANNOTATION_CONFLICT',content_sha256=digest,annotation_a=str(a),annotation_b=str(b),image_path=str(image),annotation_a_sha256=sha256(a),annotation_b_sha256=sha256(b))
    geo=dict(conflict,review_id='G-1',issue_type='GEOMETRY',annotation_b='',annotation_b_sha256='',geometry_issue=json.dumps({'shape_index':1,'self_intersection_assessment':'minor/local'}))
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([conflict,geo]))
    records=[dict(content_sha256=digest,source_annotation_path=str(p),width=16,height=16) for p in (a,b)]
    def read(c='',g='',linked=''):
        path=tmp_path/'decisions.csv'
        with path.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=FIELDS+['geometry_decisions'],extrasaction='ignore');w.writeheader()
            w.writerow(conflict|{'human_decision':c,'geometry_decisions':linked})
            if not linked: w.writerow(geo|{'human_decision':g})
        return read_decisions(path,catalog,records,tmp_path)
    return read,a,b


def test_source_selection_removes_unselected_geometry_work(overlapping_case):
    read,a,b=overlapping_case
    assert read('KEEP_B')['pending_review_ids']==[]
    assert read('KEEP_A')['pending_review_ids']==['G-1']
    assert read()['pending_review_ids']==['C-1']


def test_explicit_geometry_decisions_and_no_automatic_self_repair(overlapping_case):
    read,a,b=overlapping_case;before=a.read_bytes()
    kept=read('KEEP_A','KEEP_AS_IS')
    assert kept['pending_review_ids']==[]
    assert kept['content_plans'][0]['explicit_geometry_exceptions']==['G-1']
    for decision in ['MANUAL_FIX','AUTO_FIX_APPROVED']:
        plan=read('KEEP_A',decision)
        assert plan['pending_review_ids']==['G-1']
        assert plan['build_authorized'] is False
    assert a.read_bytes()==before
    assert read('KEEP_A',linked=json.dumps({'G-1':'KEEP_AS_IS'}))['pending_review_ids']==[]


def test_exclude_overrides_source_and_repairs(overlapping_case):
    read,a,b=overlapping_case
    for conflict,geometry in [('KEEP_A','EXCLUDE'),('EXCLUDE','AUTO_FIX_APPROVED')]:
        result=read(conflict,geometry)
        assert result['pending_review_ids']==[]
        assert result['content_plans'][0]['status']=='EXCLUDED'
        assert result['content_plans'][0]['automatic_repairs']==[]


def test_preparation_plan_uses_verified_auto_staging(tmp_path):
    image=tmp_path/'source.png';Image.new('RGB',(16,16)).save(image)
    annotation=tmp_path/'source.json'
    annotation.write_text(json.dumps(dict(imagePath=image.name,imageWidth=16,imageHeight=16,shapes=[dict(label='Tree',shape_type='polygon',points=[[1,1],[12,1],[12,1],[12,12],[1,12]])])))
    row=dict.fromkeys(FIELDS,'')
    row.update(review_id='G-auto',issue_type='GEOMETRY',content_sha256=sha256(image),annotation_a=str(annotation),annotation_a_sha256=sha256(annotation),annotation_b_sha256='',image_path=str(image),recommended_action='SAFE_REMOVE_DUPLICATE_VERTEX',geometry_issue=json.dumps({'shape_index':1}))
    records=[dict(source_annotation_path=str(annotation),source_image_path=str(image),annotation_sha256=sha256(annotation),content_sha256=sha256(image),width=16,height=16)]
    repair_cases(tmp_path,[row],records)
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([row]))
    decisions=tmp_path/'decisions.csv'
    with decisions.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerow(row)
    plan=read_decisions(decisions,catalog,records,tmp_path)
    assert plan['pending_review_ids']==[]
    assert plan['content_plans'][0]['automatic_repairs'][0]['review_id']=='G-auto'
    assert plan['content_plans'][0]['effective_generated_annotation'].endswith('generated_annotation.json')
    assert plan['build_authorized'] is False
