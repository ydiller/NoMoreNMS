# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import numpy as np
import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels


    def simple_test_deepsets(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        feats = bbox_results["last_layer_feats"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        feats = feats.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        sets_list = []
        device = cls_score[0].get_device()
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                bboxes, scores = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=None)  # skip NMS

            det_bbox_list = []
            det_labels_list = []
            sets, preds, set_labels, set_bboxes = self._forward_deepsets(bboxes, scores, feats[i],
                                                                             mode='test', ds_cfg=rcnn_test_cfg['deepsets_config'],
                                                                             img_shape=img_shapes[i], device=device, score_thr=0.05)
            for i, _set in enumerate(sets):
                pred_score = torch.nn.functional.softmax(preds[i], 0)[torch.argmax(preds[i])].item()
                pred_score *= torch.ones(1, 1).cuda(device=device)
                correct_indices = torch.argmax(preds[i])
                # correct_indices = torch.argmax(_set[:, -1])  # nms test
                bbox = set_bboxes[i][correct_indices].unsqueeze(0)
                # score = _set[correct_indices][-1] * torch.ones(1, 1).cuda(device=device)
                score = torch.max(_set[:, -1]) * torch.ones(1, 1).cuda(device=device)
                # if _set[correct_indices][-1] > torch.mean(_set[:, -1]):
                #     score = _set[correct_indices][-1] * torch.ones(1, 1).cuda(device=device)
                # else:
                #     score = torch.mean(_set[:, -1]) * torch.ones(1, 1).cuda(device=device)
                # bbox = torch.cat([bbox, score, torch.ones((len(bbox), 1)).cuda(device=device) * i], dim=1)  # added set id
                bbox = torch.cat([bbox, score, pred_score, torch.ones((len(bbox), 1)).cuda(device=device) * i], dim=1)  # added prediction
                # bbox = torch.cat([bbox, score], dim=1)
                det_bbox_list.append(bbox)
                label = set_labels[i].unsqueeze(0)
                det_labels_list.append(label)

            # statistics = sets  # rois[:, 1:]  #  # |max_score_rate
            if len(det_bbox_list) > 0:
                det_bbox = torch.stack(det_bbox_list, dim=1).squeeze()
                det_label = torch.stack(det_labels_list).squeeze()
                if det_bbox.ndim == 1:
                    det_bbox = det_bbox.unsqueeze(0)
                    det_label = det_label.unsqueeze(0)
            else:
                det_bbox = torch.zeros((1, 7), device=device)
                det_label = torch.ones((1), dtype=int, device=device)*(scores.shape[1]-1)
            k = rcnn_test_cfg.max_per_img

            _, inds = det_bbox[:, 4].sort(descending=True)  # at original file: bboxes[:, -1]
            inds = inds[:k]
            det_bbox = det_bbox[inds]
            det_label = det_label[inds]
            # if det_bbox.shape[0] < k:
            #     det_bbox = torch.cat([det_bbox, torch.zeros((k - det_bbox.shape[0]), det_bbox.shape[1]).cuda(device=device)],
            #                            0)
            #     det_label = torch.cat([det_label, torch.zeros((k - det_label.shape[0]), dtype=int).cuda(device=device)], 0)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            sets_list.append(sets)
        return det_bboxes, det_labels, sets_list

    def simple_test_deepsets_bbox(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        feats = bbox_results["last_layer_feats"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        feats = feats.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        sets_list = []
        device = cls_score[0].get_device()
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                bboxes, scores = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=None)  # skip NMS

            det_bbox_list = []
            det_labels_list = []
            sets, preds, set_labels, set_bboxes, set_scores, centroids, normalization_data = self._forward_deepsets(bboxes, scores, feats[i],
                                                                             mode='test', ds_cfg=rcnn_test_cfg['deepsets_config'],
                                                                             img_shape=img_shapes[i], device=device, score_thr=0.05)
            one = torch.ones((1, 1)).cuda(device=device)
            for j, _set in enumerate(sets):
                bbox = preds[j].T
                centroids_per_set = centroids[j]
                ndps = normalization_data[j]
                # bbox[:, 2] = (centroids_per_set[0] + (bbox[:, 2])) * img_shapes[i][1]
                # bbox[:, 3] = (centroids_per_set[1] + (bbox[:, 3])) * img_shapes[i][0]
                # bbox[:, 0] = (centroids_per_set[0] - (bbox[:, 0])) * img_shapes[i][1]
                # bbox[:, 1] = (centroids_per_set[1] - (bbox[:, 1])) * img_shapes[i][0]

                bbox[:, 2] = ((bbox[:, 2]*ndps['x2_std'])+ndps['x2_mean']) * img_shapes[i][1]
                bbox[:, 3] = ((bbox[:, 3]*ndps['y2_std'])+ndps['y2_mean']) * img_shapes[i][0]
                bbox[:, 0] = ((bbox[:, 0]*ndps['x1_std'])+ndps['x1_mean']) * img_shapes[i][1]
                bbox[:, 1] = ((bbox[:, 1]*ndps['y1_std'])+ndps['y1_mean']) * img_shapes[i][0]

                # bbox[:, 2] = ((bbox[:, 2]*0.4)+ndps['x2_mean']) * img_shapes[i][1]
                # bbox[:, 3] = ((bbox[:, 3]*0.4)+ndps['y2_mean']) * img_shapes[i][0]
                # bbox[:, 0] = ((bbox[:, 0]*0.6)+ndps['x1_mean']) * img_shapes[i][1]
                # bbox[:, 1] = ((bbox[:, 1]*0.6)+ndps['y1_mean']) * img_shapes[i][0]


                if set_bboxes[j][:, 0].sum() == set_bboxes[j][0, 0]:  # only one element in set
                    bbox[:, :4] = set_bboxes[j][torch.argmax(_set.sum(1))]
                original_set_size = sum(_set.sum(1) != 0)
                # score = torch.max(_set[:, -1]) * torch.ones(1, 1).cuda(device=device)
                score = torch.max(set_scores[j]) * torch.ones(1, 1).cuda(device=device)
                bbox = torch.cat([bbox, score, one * original_set_size, one * j], dim=1)
                det_bbox_list.append(bbox)
                label = set_labels[j].unsqueeze(0)
                det_labels_list.append(label)

            # statistics = sets  # rois[:, 1:]  #  # |max_score_rate
            if len(det_bbox_list) > 0:
                det_bbox = torch.stack(det_bbox_list, dim=1).squeeze()
                det_label = torch.stack(det_labels_list).squeeze()
                if det_bbox.ndim == 1:
                    det_bbox = det_bbox.unsqueeze(0)
                    det_label = det_label.unsqueeze(0)
            else:
                det_bbox = torch.zeros((1, 7), device=device)
                det_label = torch.ones((1), dtype=int, device=device)*(scores.shape[1]-1)
            k = rcnn_test_cfg.max_per_img

            _, inds = det_bbox[:, 4].sort(descending=True)  # at original file: bboxes[:, -1]
            inds = inds[:k]
            det_bbox = det_bbox[inds]
            det_label = det_label[inds]
            # if det_bbox.shape[0] < k:
            #     det_bbox = torch.cat([det_bbox, torch.zeros((k - det_bbox.shape[0]), det_bbox.shape[1]).cuda(device=device)],
            #                            0)
            #     det_label = torch.cat([det_label, torch.zeros((k - det_label.shape[0]), dtype=int).cuda(device=device)], 0)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            sets_list.append(set_bboxes)
        return det_bboxes, det_labels, sets_list


    def weighted_average_score(self, scores):
        scores_sum = torch.sum(scores)
        weights = scores/scores_sum
        return torch.sum(scores * weights)

    def simple_test_deepsets_bbox_valid_sets(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        feats = bbox_results["last_layer_feats"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        feats = feats.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        sets_list = []
        device = cls_score[0].get_device()
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                bboxes, scores = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=None)  # skip NMS

            det_bbox_list = []
            det_labels_list = []
            sets, preds, predicted_scores, set_labels, set_bboxes, set_scores, centroids, normalization_data = self._forward_deepsets(bboxes, scores, feats[i],
                                                                             mode='test', ds_cfg=rcnn_test_cfg['deepsets_config'],
                                                                             img_shape=img_shapes[i], device=device, score_thr=0.05)
            one = torch.ones((1, 1)).cuda(device=device)
            for j, _set in enumerate(sets):
                # bbox = preds[j][:4].T
                # set_score = preds[j][4]
                # centroids_per_set = centroids[j]
                # ndps = normalization_data[j]
                # # bbox[:, 2] = (centroids_per_set[0] + (bbox[:, 2])) * img_shapes[i][1]
                # # bbox[:, 3] = (centroids_per_set[1] + (bbox[:, 3])) * img_shapes[i][0]
                # # bbox[:, 0] = (centroids_per_set[0] - (bbox[:, 0])) * img_shapes[i][1]
                # # bbox[:, 1] = (centroids_per_set[1] - (bbox[:, 1])) * img_shapes[i][0]
                #
                # bbox[:, 2] = ((bbox[:, 2]*ndps['x2_std'])+ndps['x2_mean']) * img_shapes[i][1]
                # bbox[:, 3] = ((bbox[:, 3]*ndps['y2_std'])+ndps['y2_mean']) * img_shapes[i][0]
                # bbox[:, 0] = ((bbox[:, 0]*ndps['x1_std'])+ndps['x1_mean']) * img_shapes[i][1]
                # bbox[:, 1] = ((bbox[:, 1]*ndps['y1_std'])+ndps['y1_mean']) * img_shapes[i][0]
                # score = torch.max(set_scores[j]) * torch.ones(1, 1).cuda(device=device)

                # new architecture
                bbox = preds[j][:4]
                score = predicted_scores[j]

                if set_bboxes[j][:, 0].sum() == set_bboxes[j][0, 0]:  # only one element in set
                    bbox[:, :4] = set_bboxes[j][torch.argmax(set_scores[j])]
                    score = torch.max(set_scores[j])
                original_set_size = sum(_set.sum(1) != 0)
                # bbox[:, :4] = set_bboxes[j][torch.argmax(set_scores[j])]  # nms

                # score = self.weighted_average_score(set_scores[j]) * torch.ones(1, 1).cuda(device=device)
                # bbox = torch.cat([bbox, score, one * set_score, one * set_score], dim=1)
                bbox = torch.cat([bbox, one * score, one * score, one * score], dim=1)
                det_bbox_list.append(bbox)
                label = set_labels[j].unsqueeze(0)
                det_labels_list.append(label)

            # statistics = sets  # rois[:, 1:]  #  # |max_score_rate
            if len(det_bbox_list) > 0:
                det_bbox = torch.stack(det_bbox_list, dim=1).squeeze()
                det_label = torch.stack(det_labels_list).squeeze()
                if det_bbox.ndim == 1:
                    det_bbox = det_bbox.unsqueeze(0)
                    det_label = det_label.unsqueeze(0)
            else:
                det_bbox = torch.zeros((1, 7), device=device)
                det_label = torch.ones((1), dtype=int, device=device)*(scores.shape[1]-1)
            k = rcnn_test_cfg.max_per_img

            _, inds = det_bbox[:, 4].sort(descending=True)  # at original file: bboxes[:, -1]
            inds = inds[:k]
            det_bbox = det_bbox[inds]
            det_label = det_label[inds]
            # if det_bbox.shape[0] < k:
            #     det_bbox = torch.cat([det_bbox, torch.zeros((k - det_bbox.shape[0]), det_bbox.shape[1]).cuda(device=device)],
            #                            0)
            #     det_label = torch.cat([det_label, torch.zeros((k - det_label.shape[0]), dtype=int).cuda(device=device)], 0)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            sets_list.append(set_bboxes)
        return det_bboxes, det_labels, sets_list



    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes,
                                                    merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor,
                                              (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape,
                    scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False)
        return segm_result
