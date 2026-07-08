import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling import (
    Backbone,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
    detector_postprocess,
)
from detectron2.modeling import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class SoM_MIMO_RCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        cfg,
    ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.cfg = cfg
        self.input_format = input_format
        self.vis_period = vis_period

        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def _collect_channel_inputs(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Collect CSI branches from mapper outputs:

            h_history_wifo: [B, T_hist, H_wifo, W_wifo, 2]
            h_history_som:  [B, T_hist, Nr, Nt, 2]
            h_gt_som:       [B, T_pred, Nr, Nt, 2]
            snr_db:         [B]
        """
        h_history_wifo = None
        h_history_som = None
        h_gt_som = None
        snr_db = None

        if "h_history_wifo" in batched_inputs[0]:
            h_history_wifo = torch.stack(
                [x["h_history_wifo"] for x in batched_inputs],
                dim=0,
            ).to(self.device)

        if "h_history_som" in batched_inputs[0]:
            h_history_som = torch.stack(
                [x["h_history_som"] for x in batched_inputs],
                dim=0,
            ).to(self.device)

        if "h_gt_som" in batched_inputs[0]:
            h_gt_som = torch.stack(
                [x["h_gt_som"] for x in batched_inputs],
                dim=0,
            ).to(self.device)

        if "snr_db" in batched_inputs[0]:
            snr_db = torch.stack(
                [torch.as_tensor(x["snr_db"], dtype=torch.float32) for x in batched_inputs],
                dim=0,
            ).to(self.device)

        return h_history_wifo, h_history_som, h_gt_som, snr_db

    def visualize_training(self, batched_inputs, proposals):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input_per_image, prop in zip(batched_inputs, proposals):
            img = input_per_image["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input_per_image["instances"].gt_boxes)
            anno_img = v_gt.get_image()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes; Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        h_history_wifo, h_history_som, h_gt_som, snr_db = self._collect_channel_inputs(batched_inputs)

        # 改动点：接 4 个返回值
        features, pyramid_rx, pyramid_tx, aux_losses = self.backbone(
            images.tensor,
            h_history_wifo=h_history_wifo,
            h_history_som=h_history_som,
            h_gt_som=h_gt_som,
            snr_db=snr_db,
        )

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = self._postprocess(results, batched_inputs, images.image_sizes)

        return results, pyramid_rx, pyramid_tx

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        h_history_wifo, h_history_som, h_gt_som, snr_db = self._collect_channel_inputs(batched_inputs)

        # 改动点：接 4 个返回值
        features, pyramid_rx, pyramid_tx, aux_losses = self.backbone(
            images.tensor,
            h_history_wifo=h_history_wifo,
            h_history_som=h_history_som,
            h_gt_som=h_gt_som,
            snr_db=snr_db,
        )

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # 改动点：把 WiFo CSI 辅助损失并入总 loss
        if aux_losses is not None and len(aux_losses) > 0:
            losses.update(aux_losses)

        return losses
