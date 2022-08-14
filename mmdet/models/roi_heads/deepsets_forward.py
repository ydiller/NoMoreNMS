import torch
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


def forward_box(roi_head, multi_bboxes, cls_score, last_layer_feats, input_type,
                      mode='train', ds_cfg=None, img_shape=None, gt_labels=None, device=0, score_thr=0.05):

    num_classes = cls_score.shape[1] - 1
    preds = []
    inputs = []
    set_bboxes = []
    set_scores = []
    input_labels = []
    centroids_per_set = []
    normalization_data = []
    max_num = ds_cfg['max_num']
    iou_threshold = ds_cfg['iou_thresh']
    if mode == 'train':
        classes = torch.unique(gt_labels)
    else:
        classes = torch.range(0, num_classes - 1, dtype=int).cuda(device=device)
    zero = torch.zeros(1).cuda(device=device)
    one = torch.ones(1).cuda(device=device)
    for c in classes:  # c in 0 to data length - 1
        sets = []
        bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
        scores = cls_score[:, c]
        valid_mask = scores > score_thr
        valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores = bboxes[valid_inds], scores[valid_inds]
        scores, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        feats = last_layer_feats[inds]
        ious = bbox_overlaps(bboxes, bboxes)
        is_clustered = torch.ones(ious.shape[0]).cuda(device=device)
        for j, row in enumerate(ious):
            _set = []
            if is_clustered[j] == 1:
                selected_indices = torch.nonzero((row > iou_threshold) * is_clustered, as_tuple=False).squeeze()
                is_clustered *= torch.where(row > iou_threshold, zero, one)  # selected indices = 0
                if selected_indices.ndim == 0:
                    selected_indices = selected_indices.unsqueeze(0)
                sets.append(selected_indices)
        x1_full, x2_full, y1_full, y2_full, width_full, height_full, aspect_ratio_full, area_full, centroids_full = \
            roi_head.input_preprocessing(bboxes, img_shape, device)
        for s, _set in enumerate(sets):
            if mode == 'train' and len(_set) == 1:  # were set includes only one object
                continue
            x1 = x1_full[sets[s]]
            x2 = x2_full[sets[s]]
            y1 = y1_full[sets[s]]
            y2 = y2_full[sets[s]]
            width = width_full[sets[s]]
            height = height_full[sets[s]]
            aspect_ratio = aspect_ratio_full[sets[s]]
            area = area_full[sets[s]]
            centroids = centroids_full[sets[s]]
            set_dist_mean = torch.mean(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_dist_sum = torch.sum(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_ious_mean = torch.mean(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            set_ious_sum = torch.sum(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            current_scores = scores[sets[s]].unsqueeze(1)

            if input_type == 'bbox':
                input = torch.cat([x1, y1, x2, y2, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial':
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis':
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, feats[sets[s]], scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis_label':
                one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
                one_hot_classes[:, c] = 1
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, feats[sets[s]], one_hot_classes, scores[sets[s]].unsqueeze(1)], dim=1)
            else:
                assert False, 'Unknown input type'

            _set_bboxes = bboxes[sets[s]]
            _set_scores = scores[sets[s]].unsqueeze(1)

            randperm = torch.randperm(len(input))
            input = input[randperm]
            set_bboxes.append(_set_bboxes[randperm])
            set_scores.append(_set_scores[randperm])
            inputs.append(input)
            input_labels.append(c)
            centroids_per_set.append(torch.mean(centroids, 0))
            pred = roi_head.set_forward6(input)
            pred = pred.unsqueeze(0)
            preds.append(pred)

    if len(preds) > 0:
        preds_tensor = torch.stack(preds, dim=1).squeeze(0)
        predicted_scores = preds_tensor[:, -1]
        preds_tensor_reshaped = torch.zeros((preds_tensor.shape[0], 4)).cuda(device=device)
        preds_tensor_reshaped[:, 2] = preds_tensor[:, 2] * img_shape[1]
        preds_tensor_reshaped[:, 3] = preds_tensor[:, 3] * img_shape[0]
        preds_tensor_reshaped[:, 0] = preds_tensor[:, 0] * img_shape[1]
        preds_tensor_reshaped[:, 1] = preds_tensor[:, 1] * img_shape[0]
        input_labels = torch.stack(input_labels)
        if predicted_scores.ndim == 0:
            predicted_scores = predicted_scores.unsqueeze(0)
    else:
        predicted_scores = torch.zeros(0).cuda(device=device)
        preds_tensor_reshaped = torch.empty(0).cuda(device=device)
        input_labels = torch.empty(0).cuda(device=device)
    return inputs, preds_tensor_reshaped, predicted_scores, input_labels, set_bboxes, set_scores, centroids_per_set, normalization_data


def forward_centroids(roi_head, multi_bboxes, cls_score, last_layer_feats, input_type,
                      mode='train', ds_cfg=None, img_shape=None, gt_labels=None, device=0, score_thr=0.05):

    num_classes = cls_score.shape[1] - 1
    preds = []
    inputs = []
    set_bboxes = []
    set_scores = []
    input_labels = []
    centroids_per_set = []
    normalization_data = []
    max_num = ds_cfg['max_num']
    iou_threshold = ds_cfg['iou_thresh']
    if mode == 'train':
        classes = torch.unique(gt_labels)
    else:
        classes = torch.range(0, num_classes - 1, dtype=int).cuda(device=device)
    zero = torch.zeros(1).cuda(device=device)
    one = torch.ones(1).cuda(device=device)
    for c in classes:  # c in 0 to data length - 1
        sets = []
        bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
        scores = cls_score[:, c]
        valid_mask = scores > score_thr
        valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores = bboxes[valid_inds], scores[valid_inds]
        scores, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        feats = last_layer_feats[inds]
        ious = bbox_overlaps(bboxes, bboxes)
        is_clustered = torch.ones(ious.shape[0]).cuda(device=device)
        for j, row in enumerate(ious):
            _set = []
            if is_clustered[j] == 1:
                selected_indices = torch.nonzero((row > iou_threshold) * is_clustered, as_tuple=False).squeeze()
                is_clustered *= torch.where(row > iou_threshold, zero, one)  # selected indices = 0
                if selected_indices.ndim == 0:
                    selected_indices = selected_indices.unsqueeze(0)
                sets.append(selected_indices)
        x1_full, x2_full, y1_full, y2_full, width_full, height_full, aspect_ratio_full, area_full, centroids_full = \
            roi_head.input_preprocessing(bboxes, img_shape, device)
        for s, _set in enumerate(sets):
            if mode == 'train' and len(_set) == 1:  # were set includes only one object
                continue
            x1 = x1_full[sets[s]]
            x2 = x2_full[sets[s]]
            y1 = y1_full[sets[s]]
            y2 = y2_full[sets[s]]
            width = width_full[sets[s]]
            height = height_full[sets[s]]
            aspect_ratio = aspect_ratio_full[sets[s]]
            area = area_full[sets[s]]
            centroids = centroids_full[sets[s]]
            set_dist_mean = torch.mean(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_dist_sum = torch.sum(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_ious_mean = torch.mean(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            set_ious_sum = torch.sum(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            current_scores = scores[sets[s]].unsqueeze(1)
            # zero mean center points
            x1 = torch.mean(centroids, 0)[0] - x1
            x2 = x2 - torch.mean(centroids, 0)[0]
            y1 = torch.mean(centroids, 0)[1] - y1
            y2 = y2 - torch.mean(centroids, 0)[1]
            if input_type == 'bbox':
                input = torch.cat([x1, y1, x2, y2, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial':
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis':
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, feats[sets[s]], scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis_label':
                one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
                one_hot_classes[:, c] = 1
                input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                           set_dist_mean, set_dist_sum, feats[sets[s]], one_hot_classes, scores[sets[s]].unsqueeze(1)], dim=1)
            else:
                assert False, 'Unknown input type'

            _set_bboxes = bboxes[sets[s]]
            _set_scores = scores[sets[s]].unsqueeze(1)

            randperm = torch.randperm(len(input))
            input = input[randperm]
            set_bboxes.append(_set_bboxes[randperm])
            set_scores.append(_set_scores[randperm])
            inputs.append(input)
            input_labels.append(c)
            centroids_per_set.append(torch.mean(centroids, 0))

            pred = roi_head.set_forward6(input)
            pred = pred.unsqueeze(0)

            # de-normalization
            pred[:, 0] = torch.mean(centroids, 0)[0] - pred[:, 0]
            pred[:, 1] = torch.mean(centroids, 0)[1] - pred[:, 1]
            pred[:, 2] = torch.mean(centroids, 0)[0] + pred[:, 2]
            pred[:, 3] = torch.mean(centroids, 0)[1] + pred[:, 3]
            preds.append(pred)
    if len(preds) > 0:
        preds_tensor = torch.stack(preds, dim=1).squeeze(0)
        # predicted_scores = roi_head.set_forward5(preds_tensor)
        predicted_scores = preds_tensor[:, -1]
        preds_tensor_reshaped = torch.zeros((preds_tensor.shape[0], 4)).cuda(device=device)
        preds_tensor_reshaped[:, 2] = preds_tensor[:, 2] * img_shape[1]
        preds_tensor_reshaped[:, 3] = preds_tensor[:, 3] * img_shape[0]
        preds_tensor_reshaped[:, 0] = preds_tensor[:, 0] * img_shape[1]
        preds_tensor_reshaped[:, 1] = preds_tensor[:, 1] * img_shape[0]
        input_labels = torch.stack(input_labels)
        if predicted_scores.ndim == 0:
            predicted_scores = predicted_scores.unsqueeze(0)
    else:
        predicted_scores = torch.zeros(0).cuda(device=device)
        preds_tensor_reshaped = torch.empty(0).cuda(device=device)
        input_labels = torch.empty(0).cuda(device=device)
    return inputs, preds_tensor_reshaped, predicted_scores, input_labels, set_bboxes, set_scores, centroids_per_set, normalization_data


def forward_normalized(roi_head, multi_bboxes, cls_score, last_layer_feats, input_type,
                      mode='train', ds_cfg=None, img_shape=None, gt_labels=None, device=0, score_thr=0.05):

    num_classes = cls_score.shape[1] - 1
    preds = []
    inputs = []
    set_bboxes = []
    set_scores = []
    input_labels = []
    centroids_per_set = []
    normalization_data = []
    max_num = ds_cfg['max_num']
    iou_threshold = ds_cfg['iou_thresh']
    if mode == 'train':
        classes = torch.unique(gt_labels)
    else:
        classes = torch.range(0, num_classes - 1, dtype=int).cuda(device=device)
    zero = torch.zeros(1).cuda(device=device)
    one = torch.ones(1).cuda(device=device)
    for c in classes:  # c in 0 to data length - 1
        sets = []
        bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
        scores = cls_score[:, c]
        valid_mask = scores > score_thr
        valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores = bboxes[valid_inds], scores[valid_inds]
        scores, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        feats = last_layer_feats[inds]
        ious = bbox_overlaps(bboxes, bboxes)
        is_clustered = torch.ones(ious.shape[0]).cuda(device=device)
        for j, row in enumerate(ious):
            _set = []
            if is_clustered[j] == 1:
                selected_indices = torch.nonzero((row > iou_threshold) * is_clustered, as_tuple=False).squeeze()
                is_clustered *= torch.where(row > iou_threshold, zero, one)  # selected indices = 0
                if selected_indices.ndim == 0:
                    selected_indices = selected_indices.unsqueeze(0)
                sets.append(selected_indices)
        x1_full, x2_full, y1_full, y2_full, width_full, height_full, aspect_ratio_full, area_full, centroids_full = \
            roi_head.input_preprocessing(bboxes, img_shape, device)
        for s, _set in enumerate(sets):
            if mode == 'train' and len(_set) == 1:  # were set includes only one object
                continue
            x1 = x1_full[sets[s]]
            x2 = x2_full[sets[s]]
            y1 = y1_full[sets[s]]
            y2 = y2_full[sets[s]]
            width = width_full[sets[s]]
            height = height_full[sets[s]]
            aspect_ratio = aspect_ratio_full[sets[s]]
            area = area_full[sets[s]]
            centroids = centroids_full[sets[s]]
            set_dist_mean = torch.mean(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_dist_sum = torch.sum(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
            set_ious_mean = torch.mean(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            set_ious_sum = torch.sum(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
            current_scores = scores[sets[s]].unsqueeze(1)
            normalization_data_per_set = {
                'x1_mean': torch.mean(x1),
                'x1_std': (x1 - torch.mean(x1, 0)).norm(p=2, dim=0),  # torch.std(x1),
                'y1_mean': torch.mean(y1),
                'y1_std': (y1 - torch.mean(y1, 0)).norm(p=2, dim=0),  # torch.std(y1),
                'x2_mean': torch.mean(x2),
                'x2_std': (x2 - torch.mean(x2, 0)).norm(p=2, dim=0),  # torch.std(x2),
                'y2_mean': torch.mean(y2),
                'y2_std': (y2 - torch.mean(y2, 0)).norm(p=2, dim=0),  # torch.std(y2),
                'scores_mean': torch.mean(current_scores),
                'scores_std': (current_scores - torch.mean(current_scores, 0)).norm(p=2, dim=0),
                'set_size': len(_set),
            }

            if input_type == 'bbox':
                normalized_part = roi_head.normalize(torch.cat([x1, y1, x2, y2], dim=1))
                input = torch.cat([normalized_part, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial':
                normalized_part = roi_head.normalize([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean,
                                                      set_ious_sum, set_dist_mean, set_dist_sum], dim=1)
                input = torch.cat([normalized_part, scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis':
                normalized_part = roi_head.normalize([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean,
                                                      set_ious_sum, set_dist_mean, set_dist_sum], dim=1)
                input = torch.cat([normalized_part, feats[sets[s]], scores[sets[s]].unsqueeze(1)], dim=1)
            elif input_type == 'bbox_spacial_vis_label':
                one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
                one_hot_classes[:, c] = 1
                normalized_part = roi_head.normalize([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean,
                                                      set_ious_sum, set_dist_mean, set_dist_sum], dim=1)
                input = torch.cat([normalized_part, feats[sets[s]], one_hot_classes, scores[sets[s]].unsqueeze(1)], dim=1)
            else:
                assert False, 'Unknown input type'

            _set_bboxes = bboxes[sets[s]]
            _set_scores = scores[sets[s]].unsqueeze(1)
            randperm = torch.randperm(len(input))
            input = input[randperm]
            set_bboxes.append(_set_bboxes[randperm])
            set_scores.append(_set_scores[randperm])
            inputs.append(input)
            input_labels.append(c)
            normalization_data.append(normalization_data_per_set)
            centroids_per_set.append(torch.mean(centroids, 0))

            pred = roi_head.set_forward5(input)
            pred = pred.unsqueeze(0)

            # de-normalization
            pred[:, 0] = ((pred[:, 0] * normalization_data_per_set['x1_std']) + normalization_data_per_set[
                'x1_mean'])
            pred[:, 1] = ((pred[:, 1] * normalization_data_per_set['y1_std']) + normalization_data_per_set[
                'y1_mean'])
            pred[:, 2] = ((pred[:, 2] * normalization_data_per_set['x2_std']) + normalization_data_per_set[
                'x2_mean'])
            pred[:, 3] = ((pred[:, 3] * normalization_data_per_set['y2_std']) + normalization_data_per_set[
                'y2_mean'])
            # de-normalize score
            pred[:, 4] = ((pred[:, 4] * normalization_data_per_set['scores_std']) + normalization_data_per_set[
                'scores_mean'])

            preds.append(pred)
    if len(preds) > 0:
        preds_tensor = torch.stack(preds, dim=1).squeeze(0)
        predicted_scores = preds_tensor[:, -1]
        preds_tensor_reshaped = torch.zeros((preds_tensor.shape[0], 4)).cuda(device=device)
        preds_tensor_reshaped[:, 2] = preds_tensor[:, 2] * img_shape[1]
        preds_tensor_reshaped[:, 3] = preds_tensor[:, 3] * img_shape[0]
        preds_tensor_reshaped[:, 0] = preds_tensor[:, 0] * img_shape[1]
        preds_tensor_reshaped[:, 1] = preds_tensor[:, 1] * img_shape[0]
        input_labels = torch.stack(input_labels)
        if predicted_scores.ndim == 0:
            predicted_scores = predicted_scores.unsqueeze(0)
    else:
        predicted_scores = torch.zeros(0).cuda(device=device)
        preds_tensor_reshaped = torch.empty(0).cuda(device=device)
        input_labels = torch.empty(0).cuda(device=device)
    return inputs, preds_tensor_reshaped, predicted_scores, input_labels, set_bboxes, set_scores, centroids_per_set, normalization_data

