"""Rerun saved-best TEST inference only and export verified per-class metrics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from src.config import load_config,project_path
from src.models.mask2former_model import Mask2FormerModel


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Original test metrics.json; defaults to earliest saved test evaluation')
    args=parser.parse_args();experiment=args.experiment.resolve()
    reference=args.reference or next(iter(sorted(experiment.glob('test_*/metrics.json'))),None)
    if reference is None:raise ValueError('A saved test reference is required for aggregate verification')
    previous=json.loads(reference.read_text(encoding='utf-8'))
    cfg=load_config(experiment/'config.yaml');root=project_path(cfg,cfg['dataset']['output_dir']).resolve()
    protected=[p for folder in (root,experiment/'best',experiment/'last') for p in folder.rglob('*') if p.is_file()]
    protected += [experiment/'config.yaml',experiment/'run_identity.json',experiment/'training_history.json']
    baseline={str(p):sha(p) for p in protected}
    model=Mask2FormerModel(cfg,experiment_dir=experiment)
    metrics=model.evaluate()  # prepare(for_training=False) always selects best; only test inference.
    if any(sha(Path(p))!=h for p,h in baseline.items()):raise RuntimeError('Protected dataset/checkpoint/run input changed during evaluation')
    keys=['map50_mask','map50_95_mask','map75_mask','coco_ar100_mask']
    differences={k:metrics[k]-previous[k] for k in keys}
    if any(abs(d)>0.0005 for d in differences.values()):
        raise RuntimeError('Aggregate metrics do not reproduce the saved test evaluation; per-class summary not published. Differences: '+json.dumps(differences))
    if set(map(str,metrics['evaluation_image_ids']))!=set(map(str,previous['evaluation_image_ids'])):
        raise RuntimeError('Test image identities differ from the reference evaluation')
    table=[]
    for project_id,name in cfg['classes'].items():
        values=metrics['per_class'][name]
        table.append(dict(class_name=name,project_class_id=project_id,coco_category_id=project_id+1,AP50=values['map50_mask'],AP50_95=values['map50_95_mask'],AP75=values['map75_mask'],AR100=values['coco_ar100_mask']))
    result=dict(experiment=str(experiment),checkpoint=str(experiment/'best'),dataset=str(root),split='test',evaluation_images=metrics['evaluation_images'],metric_source=metrics['metric_source'],overall={k:metrics[k] for k in keys},per_class=table,reference_metrics=str(reference.resolve()),aggregate_differences=differences,aggregate_matches=True,dataset_and_checkpoints_unchanged=True,protected_file_sha256=baseline)
    destination=experiment/'per_class_test_metrics.json'
    temporary=destination.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(temporary,destination)
    csvpath=experiment/'per_class_test_metrics.csv';temporary=csvpath.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.replace(temporary,csvpath)
    print('\nClass | AP50 | AP50-95 | AP75 | AR100')
    for r in table:print(f"{r['class_name']} | {r['AP50']:.6f} | {r['AP50_95']:.6f} | {r['AP75']:.6f} | {r['AR100']:.6f}")
    print('Aggregate differences:',json.dumps(differences))
    print('Saved:',destination)


if __name__=='__main__':main()
