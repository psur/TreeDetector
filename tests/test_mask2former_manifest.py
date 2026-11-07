"""Dataset identity checks must follow the manifest, not a benchmark version."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from src.dataset.coco_export import dataset_fingerprint
try:
    from src.dataset.mask2former_dataset import validate_dataset
except ImportError as exc:
    raise unittest.SkipTest(f"Optional Mask2Former stack is not installed: {exc}") from exc


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);rows=[];counter=0
        for split,n in [('train',2),('val',1),('test',1)]:
            data=dict(images=[],annotations=[],categories=[{'id':1,'name':'tree'},{'id':2,'name':'TreeGroup'}])
            for _ in range(n):
                counter+=1;identity=f'image{counter}';image=self.root/split/'images'/(identity+'.png');image.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(16,16),(counter*30,20,40)).save(image)
                rows.append(dict(image_id=identity,source_image=str(image),source_annotation=identity+'.json',width=16,height=16,number_of_objects=1,split=split))
                data['images'].append(dict(id=counter,canonical_image_id=identity,file_name=image.name,width=16,height=16))
                data['annotations'].append(dict(id=counter,image_id=counter,category_id=1,iscrowd=0,segmentation=[[2,2,12,2,12,12,2,12]],bbox=[2,2,10,10],area=100))
            p=self.root/'coco/annotations'/f'instances_{split}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
        with (self.root/'dataset_manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        fingerprint=dataset_fingerprint(self.root/'dataset_manifest.csv')
        for p in (self.root/'coco/annotations').glob('*.json'):
            d=json.loads(p.read_text());d['info']={'dataset_fingerprint':fingerprint};p.write_text(json.dumps(d))

    def change_train(self,fn):
        p=self.root/'coco/annotations/instances_train.json';data=json.loads(p.read_text());fn(data);p.write_text(json.dumps(data))

    def test_default_uses_manifest_counts(self):
        report=validate_dataset(self.root)
        self.assertEqual({k:v['images'] for k,v in report['splits'].items()},{'train':2,'val':1,'test':1})

    def test_explicit_counts_are_still_enforced(self):
        validate_dataset(self.root,{'train':2,'val':1,'test':1})
        for counts in [{},{'train':115,'val':25,'test':25}]:
            with self.assertRaisesRegex(ValueError,'Expected split counts differ'): validate_dataset(self.root,counts)

    def test_equal_count_wrong_identity_is_rejected(self):
        self.change_train(lambda d:d['images'][0].update(canonical_image_id='wrong'))
        with self.assertRaisesRegex(ValueError,'identity mismatch'): validate_dataset(self.root)

    def test_duplicate_canonical_id_is_rejected(self):
        self.change_train(lambda d:d['images'][1].update(canonical_image_id='image1'))
        with self.assertRaisesRegex(ValueError,'identity mismatch'): validate_dataset(self.root)

    def test_missing_image_is_rejected(self):
        (self.root/'train/images/image1.png').unlink()
        with self.assertRaisesRegex(ValueError,'Missing or extra'): validate_dataset(self.root)

    def test_bad_filename_is_rejected(self):
        self.change_train(lambda d:d['images'][0].update(file_name='wrong.png'))
        with self.assertRaisesRegex(ValueError,'Missing or extra'): validate_dataset(self.root)

    def test_stale_fingerprint_is_rejected(self):
        self.change_train(lambda d:d['info'].update(dataset_fingerprint='stale'))
        with self.assertRaisesRegex(ValueError,'fingerprint'): validate_dataset(self.root)


if __name__=='__main__': unittest.main()
