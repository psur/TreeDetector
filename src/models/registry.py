from .yolo_model import YoloModel
def get_model(name,cfg,experiment=None):
    if name=="yolo":return YoloModel(cfg,experiment)
    if name=="detectree2":
        from .detectron2_model import Detectron2MaskRCNNModel
        return Detectron2MaskRCNNModel(cfg,experiment)
    raise ValueError(f"Unknown model: {name}")
def model_names():return ("yolo","detectree2")
