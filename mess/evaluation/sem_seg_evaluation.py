
# Extends the detectron2.evaluation.SemSegEvaluator by an additional evaluation subset (classes_of_interest)
# Additionally reports CoI-mIoU as well as specificity and sensitivity for datasets with two classes

import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import torch

try:
    from detectron2.data import MetadataCatalog
    from detectron2.utils.comm import all_gather, is_main_process, synchronize
    from detectron2.utils.file_io import PathManager
    from detectron2.evaluation import SemSegEvaluator
except:
    raise ImportError('Please install detectron2 to use the MESSSemSegEvaluator. '
                      'See https://github.com/facebookresearch/detectron2 for details.')


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
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        if hasattr(meta, 'classes_of_interest'):
            self.classes_of_interest = np.bincount(meta.classes_of_interest, minlength=self._num_classes).astype(bool)
        else:
            self.classes_of_interest = np.ones(self._num_classes).astype(bool)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean intersection-over-union averaged across classes of interest (CoI-mIoU)
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
        # Change: Consider potential no-object predictions for pos_gt
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

        res = {}
        res["mIoU"] = 100 * miou
        res["CoI-mIoU"] = 100 * coi_miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        if self._num_classes == 2:
            res["Specificity"] = 100 * acc[0]
            res["Sensitivity"] = 100 * acc[1]
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

