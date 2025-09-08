from typing import Dict, List, Optional, Tuple, Union
import logging
import datetime
import time
import os
import torch
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.evaluation import (
    COCOEvaluator,
    CityscapesInstanceEvaluator,
    DatasetEvaluator,
    inference_context,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from detectron2.structures import ImageList, Instances
from detectron2.utils.logger import log_every_n_seconds
from contextlib import ExitStack, contextmanager
from torch.cuda.amp import autocast as autocast
from detectron2.utils.comm import get_world_size, is_main_process
from PIL import Image


class MIMO_Evaluators(DatasetEvaluator):

    def __init__(self, evaluators):
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs,pyramid_rx,pyramid_tx):
        for evaluator in self._evaluators:
            if type(evaluator).__name__ == "Error_Evaluator":
                evaluator.process(features, p_rx, rx_feature, tx_feature,p_tx, source_features)
            else:
                evaluator.process(inputs,outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def get_evaluator(cfg, dataset_name, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    return MIMO_Evaluators(evaluator_list)



def inference_on_dataset(model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],repeats):
    num_devices = get_world_size()
    logger = logging.getLogger("detectron2")
    cumulative_results=OrderedDict()
    num_repeats=repeats


    for repeat_idx in range(num_repeats):
        logger.info("Start No.{} inference on {} batches".format(repeat_idx+1,len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length

        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        f={}
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs, pyramid_rx,pyramid_tx = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs, pyramid_rx, pyramid_tx)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                        name="detectron2",
                    )
                start_data_time = time.perf_counter()


        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )
        results = evaluator.evaluate()
        if results is None:
            results = {}
        logger.info("Evaluation results No.{} in csv format:".format(repeat_idx+1))
        print_csv_format(results)
         # Accumulate results
        for key, value in results.items():
            if key not in cumulative_results:
                cumulative_results[key] = []
            cumulative_results[key].append(value)
    average_results=OrderedDict()
    for key, values in cumulative_results.items():
        if isinstance(values[0], dict):
            average_results[key] = {subkey: sum(subvalues) / num_repeats for subkey, subvalues in zip(values[0].keys(), zip(*[v.values() for v in values]))}
        else:
            average_results[key] = sum(values) / num_repeats
    return average_results