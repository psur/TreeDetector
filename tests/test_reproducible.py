import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.dataset.reproducible import audit_sources, prepare_reproducible, sha256, validate_output

RULES = {'exclude_labels':['zmladenie'], 'exclude_shape_types':['circle']}


def source(root, name, color, label='tree', points=None):
    image=root/name
    image.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(16,16),color).save(image)
    data={'imagePath':image.name,'imageWidth':16,'imageHeight':16,'shapes':[{'label':label,'shape_type':'polygon','points':points or [[1,1],[12,1],[12,12],[1,12]]}]}
    image.with_suffix('.json').write_text(json.dumps(data))
    return image


def config(project, raw):
    (project/'config').mkdir(parents=True)
    (project/'config'/'dataset_sources.yaml').write_text(yaml.safe_dump({'raw_root':str(raw),**RULES}))
    return {'_project_root':str(project)}


def test_conflicts_include_excluded_variants_and_block_rebuild(tmp_path):
    raw=tmp_path/'raw'; project=tmp_path/'project'
    source(raw,'Jakub/a.png','red')
    source(raw,'Marek/b.png','red',label='zmladenie')
    for i,color in enumerate(['blue','green','yellow']): source(raw,f'Jakub/{i}.png',color)
    cfg=config(project,raw)
    existing=project/'datasets'/'benchmark_v5'; existing.mkdir(parents=True)
    (existing/'sentinel').write_text('keep')
    before={p:sha256(p) for p in raw.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='Unsafe to build'):
        prepare_reproducible(cfg,'benchmark_v5',rebuild=True)
    assert (existing/'sentinel').read_text()=='keep'
    assert before=={p:sha256(p) for p in before}
    report=json.loads((project/'results'/'benchmark_v5_audit'/'audit.json').read_text())
    assert len(report['annotation_conflicts'])==1
    assert report['annotation_conflicts'][0]['polygon_geometry_differs'] is False


def test_build_rebuild_and_validate(tmp_path):
    raw=tmp_path/'raw'; project=tmp_path/'project'
    for i in range(20): source(raw,f'Jakub/tile_{i}.png',(i*10,0,0))
    source(raw,'Marek/duplicate.png',(0,0,0),label='Tree',points=[[12,12],[12,1],[1,1],[1,12]])
    cfg=config(project,raw)
    before={p:sha256(p) for p in raw.rglob('*') if p.is_file()}
    result=prepare_reproducible(cfg,'benchmark_v5')
    assert result['included_image_count']==20
    assert result['duplicate_count']==1
    assert result['images_per_split']=={'train':14,'val':3,'test':3}
    out=project/'datasets'/'benchmark_v5'
    manifest=(out/'dataset_manifest.csv').read_bytes()
    with pytest.raises(ValueError,match='--rebuild'): prepare_reproducible(cfg,'benchmark_v5')
    prepare_reproducible(cfg,'benchmark_v5',rebuild=True)
    assert manifest==(out/'dataset_manifest.csv').read_bytes()
    assert before=={p:sha256(p) for p in before}
    label=next((out/'train'/'labels').glob('*.txt')); label.write_text('9 0 0 1 0 1 1')
    with pytest.raises(ValueError,match='invalid YOLO'): validate_output(out,20)


def test_audit_orphans_invalid_json_and_geometry(tmp_path):
    raw=tmp_path/'raw'
    a=source(raw,'Matej/a.png','red'); a.with_suffix('.json').unlink()
    b=source(raw,'Jakub/b.png','blue'); b.with_suffix('.json').write_text('{')
    source(raw,'Jakub/c.png','green',points=[[1,1],[12,12],[1,12],[12,1]])
    report,rows,_=audit_sources(raw,RULES)
    assert report['excluded_image_count']==3
    assert all(r['exclusion_reason'] for r in rows)
    assert sum(len(f['invalid_json']) for f in report['folders'])==1


def test_frozen_version(tmp_path):
    with pytest.raises(ValueError,match='frozen'):
        prepare_reproducible({'_project_root':str(tmp_path)},'benchmark_v4',rebuild=True)
