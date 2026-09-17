"""Command-line entry point."""
import argparse,json,logging
from pathlib import Path
def build_parser():
    from src.models.registry import model_names
    p=argparse.ArgumentParser(description="2D tree crown segmentation benchmark");p.add_argument("--config",default="config/config.yaml");p.add_argument("--verbose",action="store_true");sub=p.add_subparsers(dest="command",required=True);sub.add_parser("prepare",help="Validate, split, export, and summarize");sub.add_parser("info",help="Show runtime and dataset information")
    sub.add_parser("compare",help="Print benchmark results")
    prepare = sub.choices["prepare"]
    prepare.add_argument("--dataset", choices=["benchmark_v5"])
    prepare.add_argument("--rebuild", action="store_true")
    prepare.add_argument("--audit-only", action="store_true")
    prepare.add_argument("--validate-decisions", action="store_true")
    prepare.add_argument("--finalize-decisions", action="store_true", help="Validate and import completed review CSV into canonical decision history")
    prepare.add_argument("--decisions", help="Human-reviewed CSV; validates a read-only decision plan")
    smoke = sub.add_parser("smoke", help="One Mask2Former training and validation batch only")
    smoke.add_argument("--model", choices=["mask2former"], default="mask2former")
    for name in ("train","evaluate","benchmark","predict"):
        q=sub.add_parser(name);q.add_argument("--model",choices=model_names(),default="yolo");q.add_argument("--experiment",help="Existing run directory, especially for evaluate/predict")
        if name=="predict":q.add_argument("--source",help="Optional image path; omit to sample random fixed-test images")
    return p
def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        from src.config import load_config,project_path
        from src.utils.logging_utils import configure_logging
        from src.utils.reproducibility import runtime_info,set_seed
        configure_logging(args.verbose)
        cfg=load_config(args.config)
        if args.command=="prepare" and (args.validate_decisions or args.finalize_decisions):
            if args.dataset!='benchmark_v5' or args.rebuild or args.audit_only:
                raise ValueError('Decision validation/finalization requires --dataset benchmark_v5 and no --rebuild/--audit-only')
            if args.finalize_decisions:
                from src.dataset.final_decisions import finalize_decisions
                result=finalize_decisions(cfg['_project_root'],args.decisions)
            else:
                from src.dataset.final_decisions import validation_command
                result=validation_command(cfg['_project_root'],args.decisions)
            return 0 if result['safe_to_build'] else 2
        set_seed(int(cfg["training"]["seed"]));info=runtime_info();logging.info("Python %s | PyTorch %s | CUDA %s | GPU %s",info["python"],info["torch"],info["cuda"],info["gpu"])
        if args.command=="prepare":
            if args.dataset:
                from src.dataset.reproducible import prepare_reproducible
                prepare_reproducible(cfg,args.dataset,args.rebuild,args.audit_only,args.decisions)
                return 0
            if args.rebuild or args.audit_only or args.decisions:
                raise ValueError("--rebuild and --audit-only require --dataset benchmark_v5")
            from src.dataset.pipeline import prepare_dataset
            prepare_dataset(cfg)
        elif args.command=="info":
            import pandas as pd
            manifest=project_path(cfg,cfg["dataset"]["output_dir"])/"dataset_manifest.csv";counts=pd.read_csv(manifest).split.value_counts().to_dict() if manifest.exists() else {};print(json.dumps(info|{"selected_model":cfg["models"]["yolo"]["weights"],"dataset_images":sum(counts.values()),"splits":counts},indent=2))
        elif args.command=="smoke":
            from src.models.registry import get_model
            logging.info("Smoke experiment: %s", get_model(args.model, cfg).smoke_test())
        elif args.command=="compare":
            from src.benchmark.runner import compare
            compare(cfg)
        else:
            from src.benchmark.runner import run
            logging.info("Experiment: %s",run(cfg,args.command,args.model,Path(args.experiment) if args.experiment else None,getattr(args,"source",None)))
        return 0
    except ImportError as exc:logging.basicConfig(level=logging.INFO,format="%(levelname)s | %(message)s");logging.error("Missing dependency: %s. Run: pip install -r requirements.txt",exc);return 2
    except (FileNotFoundError,ValueError,RuntimeError,NotImplementedError,OSError) as exc:logging.error("%s",exc);return 2
if __name__=="__main__":raise SystemExit(main())
