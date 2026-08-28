import os,platform,random
import numpy as np
def set_seed(seed):
    os.environ["PYTHONHASHSEED"]=str(seed);random.seed(seed);np.random.seed(seed)
    try:
        import torch;torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    except ImportError:pass
def runtime_info():
    d={"python":platform.python_version(),"torch":"not installed","cuda":False,"gpu":"CPU","gpu_memory_gb":None}
    try:
        import torch;d["torch"]=torch.__version__;d["cuda"]=torch.cuda.is_available()
        if d["cuda"]:d["gpu"]=torch.cuda.get_device_name(0);d["gpu_memory_gb"]=round(torch.cuda.get_device_properties(0).total_memory/2**30,2)
    except ImportError:pass
    return d
def resolve_device(value):
    if value!="auto":return value
    try:
        import torch;return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:return "cpu"
