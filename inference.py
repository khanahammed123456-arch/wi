
from typing import Dict, List, Optional, Tuple
import logging
import datetime
import time
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import CfgNode
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    DatasetEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    print_csv_format,
    inference_context,
)
from detectron2.modeling import build_model
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.solver.build import build_lr_scheduler,build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.structures import ImageList, Instances
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.comm import get_world_size, is_main_process
from SoM_MIMO import *
from contextlib import ExitStack, contextmanager
from torch.cuda.amp import autocast,GradScaler
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler
)
from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler
from torchviz import make_dot, make_dot_from_trace

logger = logging.getLogger("detectron2")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTHONHASHSEED"] = str(3407)
# random, numpy
random.seed(3407)
np.random.seed(3407)

# torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def do_test(cfg, model):
    os.environ["PYTHONHASHSEED"] = str(3407)
    # random, numpy
    random.seed(3407)
    np.random.seed(3407)
    # torch
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator,cfg.SOLVER.EVAL_REPEAT)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_one_test(cfg, model):
    os.environ["PYTHONHASHSEED"] = str(3407)
    # random, numpy
    random.seed(3407)
    np.random.seed(3407)
    # torch
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator,1)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    add_mimo_config_city(cfg)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code

    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return do_test(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

