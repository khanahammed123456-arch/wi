# tools/visualize_som_mimo_predictions.py
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import torch
from tqdm import tqdm

# ------------------------------------------------------------
# 让脚本可以从 tools/ 目录导入项目根目录下的 train.py
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import inference_context

# 复用你原来的 train.py，保证模型、cfg、mapper、dataloader 与 eval 一致
from train import setup, Trainer


def get_instances_from_output(output):
    """
    兼容不同输出格式。

    普通 Detectron2 可能是：
        {"instances": Instances}

    你当前 SoM-MIMO 输出是：
        [{"instances": Instances}]

    所以不能直接判断 "instances" in output，
    必须先判断 output 是 list 还是 dict。
    """
    if output is None:
        return None

    # 情况 1：output 本身就是 Instances
    if hasattr(output, "has") and hasattr(output, "pred_boxes"):
        return output

    # 情况 2：output 是 dict
    if isinstance(output, dict):
        if "instances" in output:
            return output["instances"]

        # 兼容其他可能字段名
        candidate_keys = [
            "pred_instances",
            "instances_rx",
            "instances_pred",
            "results",
            "result",
        ]

        for key in candidate_keys:
            if key in output:
                value = output[key]

                if hasattr(value, "has") and hasattr(value, "pred_boxes"):
                    return value

                if isinstance(value, dict) and "instances" in value:
                    return value["instances"]

        return None

    # 情况 3：output 是 list 或 tuple
    # 你的当前输出就是 [{'instances': Instances(...)}]
    if isinstance(output, (list, tuple)):
        for item in output:
            inst = get_instances_from_output(item)
            if inst is not None:
                return inst

    return None


def filter_instances(instances, score_thresh=0.3, max_instances=50):
    """
    过滤低置信度检测结果，避免论文图过乱。
    """
    if instances is None:
        return instances

    if len(instances) == 0:
        return instances

    if instances.has("scores"):
        keep = instances.scores >= score_thresh
        instances = instances[keep]

    if len(instances) > max_instances and instances.has("scores"):
        order = torch.argsort(instances.scores, descending=True)
        instances = instances[order[:max_instances]]

    return instances


def visualize_predictions(
    cfg,
    output_dir,
    score_thresh=0.3,
    max_images=50,
    max_instances=50,
    save_empty=False,
    debug=False,
):
    """
    使用完整 Detectron2/SoM-MIMO 模型进行可视化。

    关键点：
    1. 用 Trainer.build_model(cfg) 构建完整检测模型，不是单独 backbone。
    2. 用 Trainer.build_test_loader(cfg, dataset_name) 构建 dataloader，
       这样你的 h_history_wifo / h_history_som / h_gt_som 才会被 mapper 正确加入。
    3. 用 Visualizer 绘制 outputs 里的 instances。
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("[INFO] Visualization config")
    print("[INFO] MODEL.WEIGHTS      =", cfg.MODEL.WEIGHTS)
    print("[INFO] OUTPUT_DIR         =", output_dir)
    print("[INFO] USE_WIFO           =", getattr(cfg.MODEL.MIMO, "USE_WIFO", None))
    print("[INFO] CSI_MODE           =", getattr(cfg.MODEL.MIMO, "CSI_MODE", None))
    print("[INFO] CSI_LAG            =", getattr(cfg.MODEL.MIMO, "CSI_LAG", None))
    print("[INFO] T_PRED             =", getattr(cfg.MODEL.MIMO, "T_PRED", None))
    print("[INFO] C                  =", getattr(cfg.MODEL.MIMO, "C", None))
    print("[INFO] INFER_SNR          =", getattr(cfg.MODEL.MIMO, "INFER_SNR", None))
    print("[INFO] DOPPLER_FREQ       =", getattr(cfg.MODEL.MIMO, "DOPPLER_FREQ", None))
    if hasattr(cfg.MODEL.MIMO, "USE_WIFO_ADAPTER"):
        print("[INFO] USE_WIFO_ADAPTER   =", getattr(cfg.MODEL.MIMO, "USE_WIFO_ADAPTER", None))
    print("[INFO] score_thresh       =", score_thresh)
    print("[INFO] max_images         =", max_images)
    print("[INFO] max_instances      =", max_instances)
    print("=" * 80)

    # 构建模型
    model = Trainer.build_model(cfg)
    model.eval()

    # 加载权重
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # 构建 test loader
    if len(cfg.DATASETS.TEST) == 0:
        raise ValueError("cfg.DATASETS.TEST is empty. Please check your config.")

    dataset_name = cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(dataset_name)
    data_loader = Trainer.build_test_loader(cfg, dataset_name)

    saved = 0
    visited = 0
    debug_printed = False

    with inference_context(model), torch.no_grad():
        for batch in tqdm(data_loader, desc="Visualizing"):
            visited += len(batch)

            outputs = model(batch)

            # ------------------------------------------------------------
            # 有些模型输出长度与 batch 对应：
            #   outputs = [{"instances": ...}, ...]
            # 你现在的输出是每个 output 又套了一层 list：
            #   output = [{"instances": ...}]
            # 所以下面用 zip(batch, outputs) 后再 get_instances_from_output()
            # ------------------------------------------------------------
            for data_dict, output in zip(batch, outputs):
                if saved >= max_images:
                    print(f"[DONE] Saved {saved} visualization images to {output_dir}")
                    return

                file_name = data_dict.get("file_name", None)
                if file_name is None:
                    print("[WARN] data_dict has no file_name, skip.")
                    continue

                if debug and not debug_printed:
                    print("\n[DEBUG] type(output) =", type(output))
                    if isinstance(output, dict):
                        print("[DEBUG] output keys =", list(output.keys()))
                    elif isinstance(output, (list, tuple)):
                        print("[DEBUG] output is list/tuple, len =", len(output))
                        if len(output) > 0:
                            print("[DEBUG] type(output[0]) =", type(output[0]))
                            if isinstance(output[0], dict):
                                print("[DEBUG] output[0] keys =", list(output[0].keys()))
                    else:
                        print("[DEBUG] output =", output)
                    debug_printed = True

                # 读取原图，Detectron2 read_image 默认返回 RGB
                img = read_image(file_name, format=cfg.INPUT.FORMAT)

                instances = get_instances_from_output(output)

                if instances is None:
                    print(f"[WARN] no instances found in output: {file_name}")
                    if save_empty:
                        save_empty_image(img, output_dir, saved, file_name, suffix="empty")
                        saved += 1
                    continue

                instances = instances.to("cpu")
                instances = filter_instances(
                    instances,
                    score_thresh=score_thresh,
                    max_instances=max_instances,
                )

                if len(instances) == 0:
                    print(f"[WARN] all instances filtered by score threshold: {file_name}")
                    if save_empty:
                        save_empty_image(img, output_dir, saved, file_name, suffix="filtered")
                        saved += 1
                    continue

                visualizer = Visualizer(
                    img,
                    metadata=metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE,
                )

                vis_output = visualizer.draw_instance_predictions(instances)
                vis_img = vis_output.get_image()

                base = os.path.basename(file_name)
                stem = os.path.splitext(base)[0]

                save_path = os.path.join(
                    output_dir,
                    f"{saved:04d}_{stem}_vis.png"
                )

                # vis_img 是 RGB，cv2.imwrite 需要 BGR
                cv2.imwrite(save_path, vis_img[:, :, ::-1])

                if debug:
                    print(
                        f"[SAVE] {save_path} | "
                        f"num_instances={len(instances)} | "
                        f"file={file_name}"
                    )

                saved += 1

    print(f"[DONE] Visited {visited} images.")
    print(f"[DONE] Saved {saved} visualization images to {output_dir}")


def save_empty_image(img, output_dir, idx, file_name, suffix="empty"):
    """
    没有检测结果时，也可以保存原图，方便排查。
    """
    base = os.path.basename(file_name)
    stem = os.path.splitext(base)[0]
    save_path = os.path.join(output_dir, f"{idx:04d}_{stem}_{suffix}.png")
    cv2.imwrite(save_path, img[:, :, ::-1])


def build_args():
    parser = argparse.ArgumentParser(
        description="Visualize SoM-MIMO detection predictions with Detectron2 Visualizer."
    )

    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to config yaml."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save visualization images."
    )

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Score threshold for visualization. If too few images are saved, try 0.05 or 0.1."
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to save."
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=50,
        help="Maximum number of instances drawn per image."
    )

    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Save original image even when no instances remain after filtering."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print output structure and save information."
    )

    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config options, same style as train.py."
    )

    return parser.parse_args()


def main():
    args = build_args()

    # 构造一个与 train.py setup() 兼容的 args 对象
    class SetupArgs:
        pass

    setup_args = SetupArgs()
    setup_args.config_file = args.config_file
    setup_args.opts = args.opts
    setup_args.resume = False
    setup_args.eval_only = True
    setup_args.num_gpus = 1
    setup_args.num_machines = 1
    setup_args.machine_rank = 0
    setup_args.dist_url = "tcp://127.0.0.1:49152"

    cfg = setup(setup_args)

    visualize_predictions(
        cfg=cfg,
        output_dir=args.output_dir,
        score_thresh=args.score_thresh,
        max_images=args.max_images,
        max_instances=args.max_instances,
        save_empty=args.save_empty,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()