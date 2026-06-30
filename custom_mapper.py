import copy
import hashlib
import math
import torch
import numpy as np

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from channel_generator import WiFoChannelGenerator


class WiFo_SoM_Mapper:
    """
    杈撳嚭锛?
        image:            [3, H, W]
        h_history_wifo:   [T_hist, H_wifo, W_wifo, 2]
        h_history_som:    [T_hist, Nr, Nt, 2]
        h_gt_som:         [T_pred, Nr, Nt, 2]
    """

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        mimo_cfg = cfg.MODEL.MIMO

        self.t_hist = int(mimo_cfg.T_HIST)
        self.t_pred = int(mimo_cfg.T_PRED)

        self.nr = int(mimo_cfg.Nr)
        self.nt = int(mimo_cfg.Nt)

        self.wifo_h = int(mimo_cfg.WIFO_H)
        self.wifo_w = int(mimo_cfg.WIFO_W)

        self.doppler_freq = float(mimo_cfg.DOPPLER_FREQ)
        self.infer_snr = float(mimo_cfg.INFER_SNR)

        self.train_doppler_min = float(mimo_cfg.TRAIN_DOPPLER_RANGE[0])
        self.train_doppler_max = float(mimo_cfg.TRAIN_DOPPLER_RANGE[1])

        self.train_snr_min = float(mimo_cfg.TRAIN_SNR_RANGE[0])
        self.train_snr_max = float(mimo_cfg.TRAIN_SNR_RANGE[1])

        # Joint CSI branch. WiFo consumes the high-dimensional sequence, and
        # SoM uses a low-dimensional sub-channel from the same realization.
        self.ch_gen_joint = WiFoChannelGenerator(
            T=self.t_hist + self.t_pred,
            Nr=self.wifo_h,
            Nt=self.wifo_w,
            fd=self.doppler_freq,
        )

        self._debug_print_count = 0

    def _sample_env(self, dataset_dict):
        digest = hashlib.md5(dataset_dict["file_name"].encode("utf-8")).hexdigest()
        base_seed = int(digest[:8], 16) % (2**31)

        if self.is_train:
            g = torch.Generator()
            g.manual_seed(base_seed)

            doppler_hz = float(
                torch.empty(1).uniform_(
                    self.train_doppler_min,
                    self.train_doppler_max,
                    generator=g,
                ).item()
            )
            snr_db = float(
                torch.empty(1).uniform_(
                    self.train_snr_min,
                    self.train_snr_max,
                    generator=g,
                ).item()
            )
        else:
            doppler_hz = self.doppler_freq
            snr_db = self.infer_snr

        return base_seed, doppler_hz, snr_db

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # 1) env
        base_seed, doppler_hz, snr_db = self._sample_env(dataset_dict)

        # 2) image
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # 3) CSI
        h_joint = self.ch_gen_joint.get_batch(
            batch_size=1,
            fd=doppler_hz,
            seed=base_seed,
            as_wifo_layout=False,
        ).squeeze(0)

        h_som = h_joint[:, :self.nr, :self.nt, :] / math.sqrt(self.nt)

        h_history_wifo = h_joint[:self.t_hist].to(torch.float32)
        h_history_som = h_som[:self.t_hist].to(torch.float32)
        h_gt_som = h_som[self.t_hist:self.t_hist + self.t_pred].to(torch.float32)

        dataset_dict["h_history_wifo"] = h_history_wifo
        dataset_dict["h_history_som"] = h_history_som
        dataset_dict["h_gt_som"] = h_gt_som
        dataset_dict["snr_db"] = torch.tensor(snr_db, dtype=torch.float32)

        # 4) debug
        if self._debug_print_count < 50:
            hist_last = h_history_som[-1]
            gt_now = h_gt_som[0]

            delta_mean = (hist_last - gt_now).abs().mean().item()
            hist_mean = hist_last.abs().mean().item()
            gt_mean = gt_now.abs().mean().item()

            hist_temporal_delta = 0.0
            if h_history_som.shape[0] > 1:
                hist_temporal_delta = (
                    h_history_som[1:] - h_history_som[:-1]
                ).abs().mean().item()

            print(
                f"[CSI DEBUG] is_train={self.is_train} "
                f"doppler={doppler_hz:.4f} "
                f"snr={snr_db:.4f} "
                f"hist_last_mean={hist_mean:.6f} "
                f"gt_mean={gt_mean:.6f} "
                f"delta_mean={delta_mean:.6f} "
                f"hist_temporal_delta={hist_temporal_delta:.6f}"
            )
            self._debug_print_count += 1

        # 5) annotations
        if "annotations" in dataset_dict:
            image_shape = image.shape[:2]
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

