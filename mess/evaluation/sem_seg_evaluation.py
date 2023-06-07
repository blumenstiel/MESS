
# Original class detectron2.evaluation.SemSegEvaluator
# Changes:
# Forces a prediction if models predict a no_object class
# Added report of Specificity and Sensitivity
# Added Class of Interest as an evaluation subset
# Added boundary IoU from a newer version of detectron2.SemSegEvaluator
# Added Grounding IoU as described in the paper
# Ghiasi, G. et al. (2022). Scaling open-vocabulary image segmentation with image-level labels. ECCV.

import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator

_CV2_IMPORTED = True
try:
    import cv2
except ImportError:
    _CV2_IMPORTED = False


class MESSSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        compute_grounding_iou=True,
        compute_boundary_iou=True,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        if self._num_classes == 1:
            # Add "others" class for contrastive segmentation
            self._num_classes = 2
            self.reset()
        if hasattr(meta, 'classes_of_interest'):
            self.classes_of_interest = np.bincount(meta.classes_of_interest, minlength=self._num_classes).astype(bool)
        else:
            self.classes_of_interest = np.ones(self._num_classes).astype(bool)

        self._compute_grounding_iou = compute_grounding_iou
        self._compute_boundary_iou = compute_boundary_iou
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._grounding_conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = output.to(self._cpu_device)
            # get predictions
            # Change: drop model-specific "no object" class by using only the first num_classes predictions
            pred = np.array(output[:self._num_classes].argmax(dim=0), dtype=int)

            # get ground truth
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=int)
            gt[gt == self._ignore_label] = self._num_classes

            # get grounding predictions (class names reduced to classes visible in the image)
            gt_classes = np.unique(gt)
            gt_classes = gt_classes[gt_classes != self._num_classes]
            grounding_pred = np.array(output[gt_classes].argmax(dim=0), dtype=int)
            grounding_pred = gt_classes[grounding_pred]

            # add to confusion matrix
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            # add to boundary confusion matrix
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            # add to grounding confusion matrix
            if self._compute_grounding_iou:
                self._grounding_conf_matrix += np.bincount(
                    (self._num_classes + 1) * grounding_pred.reshape(-1) + gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))
            file_path = os.path.join(self._output_dir, "conf_matrix.csv")
            np.savetxt(file_path, self._conf_matrix, delimiter=",", fmt="%d")

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        # Change: Consider the no object predictions as well
        # pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(float)[:-1]
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        coi_miou = np.sum(iou[self.classes_of_interest & acc_valid]) / np.sum(self.classes_of_interest & iou_valid)
        coi_macc = np.sum(acc[self.classes_of_interest & acc_valid]) / np.sum(self.classes_of_interest & acc_valid)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix, axis=0).astype(float)[:-1]
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]
            b_miou = np.sum(b_iou[b_iou_valid]) / np.sum(b_iou_valid)

        if self._compute_grounding_iou:
            g_iou = np.full(self._num_classes, np.nan, dtype=float)
            g_acc = np.full(self._num_classes, np.nan, dtype=float)
            g_tp = self._grounding_conf_matrix.diagonal()[:-1].astype(float)
            g_pos_gt = np.sum(self._grounding_conf_matrix, axis=0).astype(float)[:-1]
            g_pos_pred = np.sum(self._grounding_conf_matrix[:-1, :-1], axis=1).astype(float)
            g_union = g_pos_gt + g_pos_pred - g_tp
            g_iou_valid = g_union > 0
            g_iou[g_iou_valid] = g_tp[g_iou_valid] / g_union[g_iou_valid]
            g_miou = np.sum(g_iou[g_iou_valid]) / np.sum(g_iou_valid)
            g_acc_valid = pos_gt > 0
            g_acc[g_acc_valid] = g_tp[g_acc_valid] / g_pos_gt[g_acc_valid]
            g_macc = np.sum(g_acc[g_acc_valid]) / np.sum(g_acc_valid)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        if self._compute_boundary_iou:
            res["bmIoU"] = 100 * b_miou
            for i, name in enumerate(self._class_names):
                res["bIoU-{}".format(name)] = 100 * b_iou[i]
        if self._compute_grounding_iou:
            res["gmIoU"] = 100 * g_miou
            for i, name in enumerate(self._class_names):
                res["gIoU-{}".format(name)] = 100 * g_iou[i]
            res["gmACC"] = 100 * g_macc
        if self._num_classes == 2:
            res["Specificity"] = 100 * acc[0]
            res["Sensitivity"] = 100 * acc[1]
        res["CoI-mIoU"] = 100 * coi_miou
        res["CoI-mACC"] = 100 * coi_macc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name.lower()}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
