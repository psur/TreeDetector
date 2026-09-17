import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset.review import FIELDS, geometry
from src.dataset.review_decisions import read_decisions
from src.dataset.reproducible import sha256


def test_duplicate_classification_and_exact_boundary():
    square=[[0,0],[10,0],[10,10],[0,10]]
    assert geometry(square+square[:1])['category']=='A'
    assert not geometry(square+square[:1])['invalid']
    duplicate=geometry(square[:1]+square)
    assert duplicate['category']=='B' and duplicate['removal_preserves_exact_boundary']
    assert not duplicate['invalid']
    assert geometry(square+[[10,0]])['category']=='C'
    assert geometry([[0,0],[10,10],[0,10],[10,0]])['invalid']
    assert geometry([[0,0],[10,0],[5,0],[5,10]])['invalid']


@pytest.fixture
def decision_fixture(tmp_path):
    image=tmp_path/'raw.png'; Image.new('RGB',(16,16)).save(image)
    annotation=tmp_path/'raw.json'
    annotation.write_text(json.dumps(dict(imagePath=image.name,imageWidth=16,imageHeight=16,shapes=[dict(label='Tree',shape_type='polygon',points=[[1,1],[12,1],[12,12],[1,12]])])))
    digest=sha256(image)
    row=dict.fromkeys(FIELDS,''); row.update(review_id='C-test',content_sha256=digest,issue_type='ANNOTATION_CONFLICT',annotation_a=str(annotation),image_path=str(image),recommended_action='KEEP_A')
    catalog=tmp_path/'catalog.json'; catalog.write_text(json.dumps([row|{'annotation_a_sha256':sha256(annotation),'annotation_b_sha256':''}]))
    records=[dict(content_sha256=digest,source_annotation_path=str(annotation),width=16,height=16)]
    def read(decision='',selected=''):
        row.update(human_decision=decision,selected_annotation=selected)
        csvpath=tmp_path/'decisions.csv'
        with csvpath.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
        return read_decisions(csvpath,catalog,records,tmp_path)
    return read,annotation,tmp_path,digest


def test_blank_is_pending_and_keep_is_only_a_plan(decision_fixture):
    read,annotation,_,_=decision_fixture
    assert read()['pending_review_ids']==['C-test']
    assert not read()['resolved']
    assert read('KEEP_A')['selected_by_content']
    assert read('KEEP_PATH:'+str(annotation))['resolved']
    assert read('EXCLUDE')['resolved'][0]['selected_annotation']==''
    assert read('KEEP_A')['build_authorized'] is False
    with pytest.raises(ValueError,match='side does not exist'): read('KEEP_B')
    with pytest.raises(ValueError,match='Unsupported'): read('SAFE_REMOVE_DUPLICATE_VERTEX')


def test_stale_source_rejected(decision_fixture):
    read,annotation,_,_=decision_fixture
    annotation.write_text('{}')
    with pytest.raises(ValueError,match='Stale'): read('KEEP_A')


def test_fixed_copy_requires_staging_and_provenance(decision_fixture):
    read,annotation,project,digest=decision_fixture
    with pytest.raises(ValueError,match='inside data_staging'): read('GEOMETRY_FIXED_COPY',str(annotation))
    fixed=project/'data_staging/benchmark_v5/fixed.json'; fixed.parent.mkdir(parents=True)
    fixed.write_bytes(annotation.read_bytes())
    with pytest.raises(FileNotFoundError): read('GEOMETRY_FIXED_COPY',str(fixed))
    fixed.with_suffix('.provenance.json').write_text(json.dumps(dict(original_annotation=str(annotation),original_annotation_sha256=sha256(annotation),fixed_annotation_sha256=sha256(fixed),content_sha256=digest)))
    assert read('GEOMETRY_FIXED_COPY',str(fixed))['resolved'][0]['provenance']
    assert sha256(annotation)==sha256(fixed)
