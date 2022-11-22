# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def freeze_layers(model, model_type):
    if model_type == 'faster':
        for v in model.parameters():
            v.requires_grad = False
        # for v in model.backbone.parameters():
        #     v.requires_grad = False
        # for v in model.neck.parameters():
        #     v.requires_grad = False
        # for v in model.rpn_head.parameters():
        #     v.requires_grad = False
        # for v in model.roi_head.bbox_head.shared_fcs.parameters():
        #     v.requires_grad = False

        # for v in model.roi_head.ds1.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds2.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds3.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds4.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds5.parameters():
        #     v.requires_grad = True

        for v in model.roi_head.ln1.parameters():
            v.requires_grad = True
        # for v in model.roi_head.ln2.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ln3.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ln4.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ln5.parameters():
        #     v.requires_grad = True
        for v in model.roi_head.ln6.parameters():
            v.requires_grad = True
        # for v in model.roi_head.ln7.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.dropout.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.bn1.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.bn2.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.bn3.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds1.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds2.parameters():
        #     v.requires_grad = True
        # for v in model.roi_head.ds3.parameters():
        #     v.requires_grad = True

        for v in model.roi_head.set_transformer.parameters():
            v.requires_grad = True
        # for v in model.roi_head.set_transformer2.parameters():
        #     v.requires_grad = True
        return model
    elif model_type == 'retina':
        return model
    else:
        raise ValueError