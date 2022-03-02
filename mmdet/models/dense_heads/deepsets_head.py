import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from ..builder import HEADS, build_loss
from ..losses import accuracy
# from mmdet.core.bbox.geometry import bbox_overlaps
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


DS_TYPE = 'hard'  # soft | hard
class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = self.Gamma.weight
        self.bias = self.Gamma.bias


    def forward(self, x):
        # if x.ndim == 1:
        #     x = x.unsqueeze(0)
        xm = x.mean(0, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        # return self.Gamma(x) - self.Lambda(xm)
        return x


class SoftCrossEntropyLoss():
   def __init__(self):
      super().__init__()
      # self.weights = weights

   def forward(self, y_hat, y):
      p = F.log_softmax(y_hat, 0)
      # w_labels = self.weights*y
      loss = -(y*p.T).mean()
      return loss


@HEADS.register_module
class DeepsetsHead(nn.Module):
    def __init__(self,
                 indim=1033,  # 2260 | 8 for no features
                 ds1=1000,  # previously 400
                 ds2=600,  # previously 150
                 ds3=300,  # previously 50
                 set_size=6,
                 reg=5,
                 include_ds4=0,
                 loss_mse=dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 loss_ce=dict(
                     type='CrossEntropyLoss',
                     loss_weight=1.0)):
        super(DeepsetsHead, self).__init__()
        self.set_size = set_size
        self.reg = reg
        self.include_ds4 = include_ds4
        self.loss_mse = build_loss(loss_mse)
        self.loss_ce = build_loss(loss_ce)
        self.ds1 = PermEqui2_mean(indim, ds1)
        self.ds2 = PermEqui2_mean(ds1, ds2)
        self.ds3 = PermEqui2_mean(ds2, ds3)
        self.bn1 = torch.nn.BatchNorm1d(ds1)
        self.bn2 = torch.nn.BatchNorm1d(ds2)
        self.bn3 = torch.nn.BatchNorm1d(ds3)
        if not include_ds4:
            self.ds4 = PermEqui2_mean(ds3, 1)
        else:
            self.ds4 = PermEqui2_mean(ds3, int(ds3/2))
            self.bn4 = torch.nn.BatchNorm1d(int(ds3/2))
            self.ds5 = PermEqui2_mean(int(ds3/2), 1)



    def set_forward(self, x):
        x = F.elu(self.ds1(x))
        x = F.elu(self.ds2(x))
        x = F.elu(self.ds3(x))
        if not self.include_ds4:
            pred = F.elu(self.ds4(x))
        else:
            x = self.bn4(F.elu(self.ds4(x)))
            pred = F.elu(self.ds5(x))
        return pred


    def my_clustering(self, ious, iou_threshold):
        # sets = []
        idx = np.empty(len(ious), dtype=int)
        c = 0
        is_clustered = torch.ones(len(ious), requires_grad=True).cuda()
        for j, row in enumerate(ious):
            if is_clustered[j] == 1:
                selected_indices = torch.nonzero((row > iou_threshold) * is_clustered).squeeze()
                is_clustered *= torch.where(row > iou_threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())
                # if selected_indices.dim() > 0:
                idx[selected_indices.cpu().numpy()] = c
                # sets.append(selected_indices)
                c += 1
        return idx


    def forward(self, multi_bboxes, cls_score, last_layer_feats,
                mode='train', ds_cfg=None, img_shape=None, gt_labels=None):
        """
        Args:
            labels: bbox labels (assigned by comparing to gt)
        """
        preds = []
        inputs = []
        set_bboxes = []
        input_labels = []
        top_c = ds_cfg['top_c']
        max_num = ds_cfg['max_num']
        iou_threshold = ds_cfg['iou_thresh']
        if mode == 'train':
            classes = torch.unique(gt_labels)
        else:
            classes = torch.unique(torch.topk(cls_score[:, 1:], top_c, dim=1)[1]) + 1  # top_classes_on_set
        for c in classes:
            sets = []
            bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
            scores = cls_score[:, c]
            scores, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            feats = last_layer_feats[inds]
            ious = bbox_overlaps(bboxes, bboxes)
            is_clustered = torch.ones(ious.shape[0]).cuda()
            for j, row in enumerate(ious):
                if is_clustered[j] == 1:
                    selected_indices = torch.nonzero((row > iou_threshold) * is_clustered).squeeze()
                    is_clustered *= torch.where(row > iou_threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())
                    sets.append(selected_indices)

            for s, _set in enumerate(sets):
                if _set.ndim == 0:  # were set includes only one object
                    continue
                else:
                    x1 = bboxes[sets[s], 0].unsqueeze(1)/img_shape[1]
                    x2 = bboxes[sets[s], 2].unsqueeze(1)/img_shape[1]
                    y1 = bboxes[sets[s], 1].unsqueeze(1)/img_shape[0]
                    y2 = bboxes[sets[s], 3].unsqueeze(1)/img_shape[0]
                    width = (x2 - x1)/img_shape[1]
                    height = (y2 - y1)/img_shape[0]
                    aspect_ratio = torch.div(width, height+sys.float_info.epsilon)
                    area = width*height
                    # normalize?
                    # boxes = bboxes[sets[s]]
                    # width = (boxes[:, 2] - boxes[:, 0])
                    # height = (boxes[:, 3] - boxes[:, 1])
                    # boxes = (boxes - torch.mean(boxes, 0)) / (torch.var(boxes, 0) + sys.float_info.epsilon)
                    # width = ((width - torch.mean(width)) / torch.var(width) + sys.float_info.epsilon).unsqueeze(1)
                    # height = ((height - torch.mean(height)) / torch.var(height) + sys.float_info.epsilon).unsqueeze(1)
                    input = torch.cat([bboxes[sets[s]], feats[sets[s]], width, height, aspect_ratio, area,
                                       scores[sets[s]].unsqueeze(1)], dim=1)
                    _set_bboxes = bboxes[sets[s]]
                    # if len(input) < self.set_size:
                    #     # zero padding
                    #     input = torch.cat([input, torch.zeros((self.set_size - input.shape[0]), input.shape[1]).cuda()], 0)
                    #     _set_bboxes = torch.cat([_set_bboxes, torch.zeros((self.set_size - _set_bboxes.shape[0]), _set_bboxes.shape[1]).cuda()], 0)
                    # else:
                    _, top_scores_idx = input[:, -1].sort(descending=True)
                    # top_scores_idx = top_scores_idx[:self.set_size]
                    input = input[top_scores_idx]
                    _set_bboxes = _set_bboxes[top_scores_idx]
                    randperm = torch.randperm(len(input))
                    input = input[randperm]
                    set_bboxes.append(_set_bboxes[randperm])
                    inputs.append(input)
                    input_labels.append(c)
                    pred = self.set_forward(input) # nms testing
                    preds.append(pred)
        return inputs, preds, input_labels, set_bboxes

    def get_target(self,
                   sets,
                   set_labels,
                   set_bboxes,
                   gt_bboxes,
                   gt_labels,
                   preds):
        """
        returns deepsets labels.
        a. for each class on each set, find closest gt,
        b. calculate iou between set and gt box,
        c. normalize with max iou.
        """
        one_hot_targets = []
        soft_targets = []
        valid_preds = []
        valid_set_bboxes = []  # for wandb log
        valid_set_scores = []  # for wandb log
        valid_ious = []  # for error viz
        for j, set in enumerate(sets):
            c = set_labels[j]
            # c = torch.argmax(set[0, -1231:]).item()
            if len(set) > 0 and c in gt_labels:
                # gt_class_inds = torch.tensor(np.where(gt_labels.cpu().numpy() == c)).cuda().squeeze()
                gt_class_inds = torch.where(gt_labels == c)[0].squeeze()
                # takes boxes of relevant gt
                class_boxes = torch.index_select(gt_bboxes, 0, gt_class_inds)
                gt_iou = bbox_overlaps(set_bboxes[j], class_boxes)
                vals, row_idx = gt_iou.max(0)  # row idx: indices of set elements with max ious with each GT (1, num_gts)
                col_idx = vals.argmax(0)  # col idx: index of GT with highest iou with any set element
                # min max norm
                # min_vals, min_row_idx = gt_iou[:, col_idx].min(0)
                # v_max = gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon
                # v_min = gt_iou[min_row_idx, col_idx]+sys.float_info.epsilon
                # if v_max == v_min:
                #     v_min = 0
                # if DS_TYPE == 'soft':
                    # targets.append((gt_iou[:, col_idx] - v_min) / (v_max - v_min) * (1.0 - 0.0) + 0.0 +sys.float_info.epsilon)

                if torch.max(gt_iou[:, col_idx]) < 0.5:
                    continue
                soft_targets.append(gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon))
                # soft_targets.append(gt_iou[:, col_idx] / torch.sum(gt_iou[:, col_idx]))
                # soft_targets.append(torch.pow(gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx] + sys.float_info.epsilon), 3))
                # soft_targets.append(gt_iou[:, col_idx])
                # t = gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon)
                # t -= t.min(0, keepdim=True)[0]
                # t /= t.max(0, keepdim=True)[0]
                # if torch.sum(torch.isnan(t)) == 0:
                #     soft_targets.append(t)
                # else:
                #     soft_targets.append(torch.ones_like(t))
                # else:
                trg = torch.zeros(len(set), dtype=torch.int64).cuda()
                trg[row_idx[col_idx]] = 1
                one_hot_targets.append(trg)
                valid_preds.append(preds[j])
                valid_set_bboxes.append(set_bboxes[j])
                valid_set_scores.append(sets[j][:, -1])
                valid_ious.append(gt_iou[:, col_idx])
                    # soft x-ent
                    # trg = gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx] + sys.float_info.epsilon)
                    # trg = torch.softmax(trg, 0)
                    # targets.append(trg)

            # else:
            #     trg = torch.zeros(len(set)).cuda()
            #     targets.append(trg)

        return valid_preds, valid_set_bboxes, valid_set_scores, valid_ious, one_hot_targets, soft_targets

    def loss(self, preds, valid_ious, one_hot_targets, soft_targets, valid_set_scores):
        losses = dict()
        c = sys.float_info.epsilon
        skipped = 0
        if preds is not None:
            _mse_loss = torch.zeros(1).cuda()
            _ce_loss = torch.zeros(1).cuda()
            _soft_ce = torch.zeros(1).cuda()
            _error = torch.zeros(1).cuda()
            _ds_acc = torch.zeros(1).cuda()
            iou_error = torch.zeros(1).cuda()
            _ds_pred_on_max = torch.zeros(1).cuda()
            for i, pred in enumerate(preds):
                # if torch.sum(one_hot_targets[i]) == 0:
                #     skipped += 1
                #     print('prob')
                #     continue
                # if DS_TYPE == 'hard':
                    # weight = ((deepsets_targets[i]) * (len(deepsets_targets[i][deepsets_targets[i] == 0])-1) + 1).type(torch.float)
                    # weight = ((deepsets_targets[i]) * 9 + 1).type(torch.float)  # const wegiht
                    # _loss += self.loss_ce(  # self.loss_ms
                    #     pred,  # pred.squeeze()
                    #     deepsets_targets[i],
                    #     weight=weight)
                _ce_loss += self.loss_ce(  # self.loss_ms
                    pred.T,  # pred.squeeze()
                    torch.argmax(one_hot_targets[i]).unsqueeze(0))
                _ds_acc += torch.argmax(pred) == torch.argmax(one_hot_targets[i])
                iou_error += torch.max(valid_ious[i]) - valid_ious[i][torch.argmax(pred)]

                #nms testing:
                # _ds_acc += torch.argmax(valid_set_scores[i]) == torch.argmax(one_hot_targets[i])
                # iou_error += torch.max(valid_ious[i]) - valid_ious[i][torch.argmax(valid_set_scores[i])]

                # else:
                # _mse_loss += self.loss_mse(  # self.loss_ms
                #     pred,  # pred.squeeze()
                #     soft_targets[i])
                    # _ds_acc += torch.argmax(pred) == torch.argmax(deepsets_targets[i])
                    # _ds_pred_on_max += pred[torch.argmax(deepsets_targets[i])]
                # _error += torch.sum(torch.abs(deepsets_targets[i]-pred.squeeze()))/pred.shape[0]
                # _ce_loss += SoftCrossEntropyLoss().forward(pred, soft_targets[i])
                c += 1
            _ce_loss /= c
            # _mse_loss /= c
            # _soft_ce /= c
            _error /= c
            _ds_acc /= c
            iou_error /= c
            _ds_pred_on_max /= c
        if len(preds) == 0:
            print('prob')
            dl = torch.zeros(1).cuda()
            dl.requires_grad = True
            losses['loss_deepsets_ce'] = dl
            # losses['loss_deepsets_mse'] = dl
            losses['ds_acc'] = torch.zeros(1).cuda()
            losses['iou_error'] = torch.zeros(1).cuda()
        else:
            losses['loss_deepsets_ce'] = _ce_loss + self.reg*iou_error
            # losses['loss_deepsets_mse'] = _mse_loss
            losses['ds_acc'] = _ds_acc
            losses['iou_error'] = iou_error
        # losses['ds_pred_on_max'] = _ds_pred_on_max
        return losses
