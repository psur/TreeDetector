import pandas as pd
import pytest
from src.dataset.coco_export import polygon_to_coco,dataset_fingerprint
from src.models.registry import model_names

def test_polygon_conversion():
 seg,box,area=polygon_to_coco([(0,0),(10,0),(10,5),(0,5)],10,5)
 assert seg==[0,0,10,0,10,5,0,5] and box==[0,0,10,5] and area==50

@pytest.mark.parametrize("points",[[(0,0),(1,1)],[(0,0),(1,1),(2,2)],[(0,0),(11,0),(0,1)]])
def test_bad_polygons(points):
 with pytest.raises(ValueError):polygon_to_coco(points,10,10)

def test_fingerprint_is_row_order_independent(tmp_path):
 rows=[{"image_id":"a","split":"train","source_image":"a.png","width":10,"height":10},{"image_id":"b","split":"test","source_image":"b.png","width":10,"height":10}]
 a=tmp_path/"a.csv";b=tmp_path/"b.csv";pd.DataFrame(rows).to_csv(a,index=False);pd.DataFrame(reversed(rows)).to_csv(b,index=False)
 assert dataset_fingerprint(a)==dataset_fingerprint(b)

def test_registry():assert model_names()==("yolo","detectree2")
