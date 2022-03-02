# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
import wandb
import mmcv
import torch
import torch.distributed as dist
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


STOP_TEST_AT = 1


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    viz_dir=None,
                    show_viz=0,
                    with_wandb=0):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # if i >= STOP_TEST_AT:     # temporary condition for testing
        #     break
        with torch.no_grad():
            result, bboxes, labels, sets = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if i == 0:
            const_batch_size = batch_size
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
        if show_viz:
            for j in range(batch_size):
                idx = i*const_batch_size+j
                img_metas = data['img_metas'][0].data[0]
                # clean_dir(imgs_dir)  # clean dest dir cintent
                img_id = dataset.img_ids[idx]
                gt = dataset.get_ann_info(idx)
                gt_dict = dict()
                gt_dict['id'] = img_id
                gt_dict['bboxes'] = gt['bboxes']
                gt_dict['labels'] = gt['labels']
                gt_dict['classes'] = dataset.CLASSES
                scale_factor = img_metas[j]['scale_factor']
                img = data['img'][0].data[0][j].numpy()  # get the image
                img -= img.min()
                img /= img.max()
                img *= 255
                img = img.astype(np.uint8)
                img = np.moveaxis(img, 0, -1)
                img = cv.UMat(img).get()
                h, w, _ = img_metas[j]['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_metas[j]['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                # nms error
                # getIouErrorPerObject(img_show, bboxes[j], labels[j], gt_dict, img_id=idx, out_dir=viz_dir)
                # ds error
                getIouErrorPerObjectWithSetDrawing(img_show, bboxes[j], labels[j], sets[j], gt_dict, scale_factor, img_id=idx, out_dir=viz_dir)

        if with_wandb:
            for j in range(batch_size):
                idx = i * batch_size + j
                img_metas = data['img_metas'][0].data[0]
                img = data['img'][0].data[0][j].numpy()  # get the image
                img -= img.min()
                img /= img.max()
                img *= 255
                img = img.astype(np.uint8)
                img = np.moveaxis(img, 0, -1)
                img = cv.UMat(img).get()
                h, w, _ = img_metas[j]['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_metas[j]['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                log_boxes(idx, dataset, bboxes[j], labels[j], img_show)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def getIouErrorPerObjectWithSetDrawing(img, bboxes, labels, sets, gt_dict, scale_factor, img_id, out_dir=None):
    """
    Use for Deepsets visualization.
    For every GT, calculate IoU with bboxes of same class.
    The IoU-Error is 1-highest IoU.
    Normalize total error by number of GT.
    Drawing (white) rectangles of sets which the prediction chosen from.

    args:
        set_idx: indices of sets which the bboxes chosen from (300,1)
    """
    error = 0
    errors = []
    inds = []
    valid_sets = []
    maxdets = 100
    device = bboxes.get_device()
    empty_box = torch.zeros(1, 7).cuda(device=device)
    bboxes = torch.cat((bboxes, empty_box), 0)
    labels = torch.cat((labels, torch.zeros(1, dtype=int).cuda(device=device)), 0)
    # scores = np.concatenate((scores, np.zeros((1, 1))), 0)
    gt_boxes = torch.from_numpy(gt_dict['bboxes']).float().cuda(device=device)
    gt_labels = torch.from_numpy(gt_dict['labels']).long().cuda(device=device)
    classes_names = gt_dict['classes']
    for i, g in enumerate(gt_boxes):
        gt_class = gt_labels[i]
        preds_iou = bbox_overlaps(g.unsqueeze(0), bboxes[labels == gt_class][:, :4])
        # iou_before_nms = bbox_overlaps(g.unsqueeze(0), multi_bboxes[:, gt_class * 4:(gt_class + 1) * 4])
        if torch.numel(preds_iou) != 0:
            if torch.max(preds_iou) > 0:
                error += 1-torch.max(preds_iou)
                bbox_idx = torch.where(labels == gt_class)[0][torch.argmax(preds_iou)]
                inds.append(bbox_idx)  # bbox index
                errors.append(1 - torch.max(preds_iou))
                valid_sets.append(sets[int(bboxes[bbox_idx][-1].item())][:, :4])
            else:
                inds.append(len(bboxes) - 1)  # empty bbox index
                errors.append(torch.ones(1).cuda(device=device))
                valid_sets.append(empty_box[:4])

        else:  # no prediction matching this GT
            inds.append(len(bboxes)-1)  # empty bbox index
            errors.append(torch.ones(1).cuda(device=device))
            valid_sets.append(empty_box[:4])
    if len(gt_boxes)>0:
        error /= len(gt_boxes)

    for j in range(len(gt_boxes)):
        img_to_save = img.copy()
        for b in valid_sets[j]:
            cv.rectangle(img_to_save, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
        # gt
        x1 = int(gt_boxes[j][0])
        y1 = int(gt_boxes[j][1])
        x2 = int(gt_boxes[j][2])
        y2 = int(gt_boxes[j][3])
        class_name = classes_names[gt_labels[j]]
        color = (0, 255, 0)  # green for gt
        # score = str(format(scores[j], '.2f'))
        cv.rectangle(img_to_save, (x1, y1), (x2, y2), color, 1)
        cv.putText(img_to_save, str(class_name), (x1, y1-15), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0),
                   thickness=1)  # Write class number

        # prediction
        x1 = int(bboxes[inds[j]][0])
        y1 = int(bboxes[inds[j]][1])
        x2 = int(bboxes[inds[j]][2])
        y2 = int(bboxes[inds[j]][3])
        current_error = errors[j]
        if inds[j] < maxdets:
            color = (255, 0, 0)  # red for prediction
            p = 'pos'
        else:
            color = (255, 165, 0)
            p = 'miss'
        cv.rectangle(img_to_save, (x1, y1), (x2, y2), color, 1)  # Draw Rectangle with the coordinates
        # write class labels
        cv.putText(img_to_save,  f'e: {current_error.item():.2f} s: {bboxes[inds[j]][4]:.2f} p: {bboxes[inds[j]][5]:.2f}', (x1, y1-3), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0),
                   thickness=3)  # stroke
        cv.putText(img_to_save, f'e: {current_error.item():.2f} s: {bboxes[inds[j]][4]:.2f} p: {bboxes[inds[j]][5]:.2f}', (x1, y1-3), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                   thickness=1)

        plt.imsave(f'{out_dir}/i{img_id}_g{j}_{p}_{current_error.item():.2f}.png', img_to_save)
        # save bboxes as csv:
        df1 = pd.DataFrame(bboxes[:, :-2].cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2', 'score'])
        df2 = pd.DataFrame([classes_names[int(x.item())] for x in labels], columns=['class'])
        bboxes_df = pd.concat([df1, df2], 1)
        bboxes_df.to_csv(f'results/topk_bboxes_csv/bboxes_{img_id}.csv')

    return img, error


def getIouErrorPerObject(img, bboxes_mat, labels_mat, gt_dict, img_id, out_dir=None):
    """
    Use for NMS visualization.
    For every GT, calculate IoU with bboxes of same class.
    The IoU-Error is 1-highest IoU.
    Normalize total error by number of GT.
    Drawing (white) rectangles of all predictions with some iou with GT
    """
    error = 0
    errors = []
    inds = []
    boxes_before_nms = []
    maxdets = 100
    device = bboxes_mat[0].get_device()
    bboxes_mat = torch.cat((bboxes_mat, torch.zeros(1, 5).cuda(device=device)), 0)
    labels_mat = torch.cat((labels_mat, torch.zeros(1,dtype=int).cuda(device=device)), 0)
    # scores = np.concatenate((scores, np.zeros((1, 1))), 0)
    gt_boxes = torch.from_numpy(gt_dict['bboxes']).float().cuda(device=device)
    gt_labels = torch.from_numpy(gt_dict['labels']).long().cuda(device=device)
    classes_names = gt_dict['classes']
    for i, g in enumerate(gt_boxes):
        gt_class = gt_labels[i]
        iou = bbox_overlaps(g.unsqueeze(0), bboxes_mat[labels_mat == gt_class][:, :4])
        # iou_bofore_nms = bbox_overlaps(g.unsqueeze(0), multi_bboxes[:, gt_class*4:(gt_class+1)*4])
        if torch.numel(iou) != 0:
            if torch.max(iou) > 0.01:
                error += 1-torch.max(iou)
                inds.append(torch.where(labels_mat == gt_class)[0][torch.argmax(iou)])  # bbox index
                errors.append(1 - torch.max(iou))
                # boxes_before_nms.append(multi_bboxes[(iou_bofore_nms > 0.5).squeeze(), gt_class * 4:(gt_class + 1) * 4])
            else:
                inds.append(len(bboxes_mat) - 1)  # bbox index
                errors.append(torch.ones(1).cuda())
                # try:
                #     boxes_before_nms.append(
                #         multi_bboxes[(iou_bofore_nms > 0.5).squeeze(), gt_class * 4:(gt_class + 1) * 4])
                # except:
                #     boxes_before_nms.append(torch.zeros(1).cuda(device=device))
        else:
            inds.append(len(bboxes_mat)-1)  # bbox index
            errors.append(torch.ones(1).cuda(device=device))
            # try:
            #     boxes_before_nms.append(multi_bboxes[(iou_bofore_nms > 0.5).squeeze(), gt_class * 4:(gt_class + 1) * 4])
            # except:
            #     boxes_before_nms.append(torch.zeros(1).cuda(device=device))
    if len(gt_boxes) >0:
        error /= len(gt_boxes)

    for j in range(len(inds)):
        img_to_save = img.copy()
        # for b in boxes_before_nms[j]:
        #     cv.rectangle(img_to_save, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
        # gt
        x1 = int(gt_boxes[j][0])
        y1 = int(gt_boxes[j][1])
        x2 = int(gt_boxes[j][2])
        y2 = int(gt_boxes[j][3])
        class_name = classes_names[gt_labels[j]]
        color = (0, 255, 0)  # green for gt
        # score = str(format(scores[j], '.2f'))
        cv.rectangle(img_to_save, (x1, y1), (x2, y2), color, 1)
        cv.putText(img_to_save, str(class_name), (x1, y1-15), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0),
                   thickness=1)  # Write class number

        # prediction
        x1 = int(bboxes_mat[inds[j]][0])
        y1 = int(bboxes_mat[inds[j]][1])
        x2 = int(bboxes_mat[inds[j]][2])
        y2 = int(bboxes_mat[inds[j]][3])
        current_error = errors[j]
        if inds[j] < maxdets:
            color = (255, 0, 0)  # red for prediction
            p = 'pos'
        else:
            color = (255, 165, 0)
            p = 'miss'
        cv.rectangle(img_to_save, (x1, y1), (x2, y2), color, 1)  # Draw Rectangle with the coordinates
        # write class labels
        cv.putText(img_to_save,  f'e: {current_error.item():.2f} s:{bboxes_mat[inds[j]][4]:.2f} r: {inds[j]}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0),
                   thickness=3)  # stroke
        cv.putText(img_to_save, f'e: {current_error.item():.2f} s:{bboxes_mat[inds[j]][4]:.2f} r: {inds[j]}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                   thickness=1)
        plt.imsave(f'{out_dir}/i{img_id}_g{j}_{p}_{current_error.item():.2f}.png', img_to_save)


    return img, error


def log_boxes(index, dataset, det_bboxes, det_labels, img):
    gt = dataset.get_ann_info(index)  # 'bboxes', 'labels'
    classes_list = dataset.CLASSES
    class_id_to_label = {k: v for k, v in enumerate(classes_list)}
    pred_boxes_log = []
    gt_boxes_log = []
    proposal_log = []

    for i, box in enumerate(det_bboxes):
        class_name = classes_list[det_labels[i]]
        if (det_labels[i]).item() not in gt['labels']:
            continue
        box_data = {
            "position": {
                "minX": box[0].item(),
                "maxX": box[2].item(),
                "minY": box[1].item(),
                "maxY": box[3].item()},
            "class_id": det_labels[i].item(),
            "domain": "pixel",
            "box_caption": f'{class_name} {box[4].item():.3f} {i}',
            "scores": {"score": box[4].item(),
                       "rank": i,
                       "set id": box[6].item()}
        }
        pred_boxes_log.append(box_data)

    for _b, b in enumerate(gt['bboxes']):
        class_name = classes_list[gt['labels'][_b].item()]
        box_data = {
            "position": {
                "minX": b[0].item(),
                "maxX": b[2].item(),
                "minY": b[1].item(),
                "maxY": b[3].item()},
            "class_id": gt['labels'][_b].item(),
            "domain": "pixel",
            "box_caption": f'{class_name}'
        }
        gt_boxes_log.append(box_data)

    # for i, box in enumerate(statistics):
    #     box_data = {
    #         "position": {
    #             "minX": box[0].item(),
    #             "maxX": box[2].item(),
    #             "minY": box[1].item(),
    #             "maxY": box[3].item()},
    #         "class_id": i,
    #         "domain": "pixel",
    #         "box_caption": f'{i}',
    #         "scores": {"proposal id": i}
    #     }
    #     proposal_log.append(box_data)

    wandb.log({ "image": wandb.Image(img,
          boxes={"predictions": {"box_data": pred_boxes_log, "class_labels": class_id_to_label},
                 "gts": {"box_data": gt_boxes_log, "class_labels": class_id_to_label}
                 })})