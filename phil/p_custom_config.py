from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import vgg_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset
import os
import sys
import logging
import argparse
import itertools
import torch
import inspect
import json

import numpy as np
from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

# relative path hack
test_path = os.path.join(os.path.dirname(__file__), '..', 'Common')
test = os.path.join(sys.path[0], '..')
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
python_ssd_traindir = os.path.join(
    parentdir, 'python', 'training', 'detection', 'ssd')
sys.path.append(python_ssd_traindir)



def get_create_net(model_name,model_params):
    if model_name == 'vgg16-ssd':
        create_net = create_vgg_ssd
        def create_net(num): return create_vgg_ssd(
            num, **model_params)
    elif model_name == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        def create_net(num): return create_mobilenetv1_ssd(
            num, **model_params)
    elif model_name == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        def create_net(num): return create_mobilenetv1_ssd_lite(
            num, **model_params)
    elif model_name == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        def create_net(num): return create_squeezenet_ssd_lite(
            num, **model_params)
    elif model_name == 'mb2-ssd-lite':
        def create_net(num): return create_mobilenetv2_ssd_lite(
            num, **model_params)
    else:
        raise Exception("The net type is wrong.")

    return create_net

def get_model_fns(json_path):
    json_dict = json.load(open(json_path, 'r'))
    config_dict = json_dict['config']
    config = CustomConfig.from_json_dict(config_dict)
    model_name = json_dict["net"]
    model_params = json_dict["model_params"]
    create_net = get_create_net(model_name,model_params)
    return create_net,config,model_name

class CustomConfig(object):
    def __init__(self, image_size, image_mean, image_std,
                 iou_threshold, center_variance, size_variance,
                 specs, priors):
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.iou_threshold = iou_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.specs = specs
        self.priors = priors

    @classmethod
    def from_json_dict(cls, json_dict):
        image_size = json_dict['image_size']
        image_mean = np.array(json_dict['image_mean'])
        spec_config = json_dict['specs']
        specs = []
        for spec_info in spec_config:
            if isinstance(spec_info, dict):
                spec = SSDSpec(spec_info['feature_map_size'],
                               spec_info['shrinkage'],
                               SSDBoxSizes(*spec_info['box_sizes']),
                               spec_info['aspect_ratios'])
            elif isinstance(spec_info, list):
                if not isinstance(spec_info[2], list) or not isinstance(spec_info[2], list):
                    raise ValueError('specs[2] and specs[3] must be lists')
                spec = SSDSpec(spec_info[0],
                               spec_info[1],
                               SSDBoxSizes(*spec_info[2]),
                               spec_info[3])
            else:
                raise ValueError(
                    'spec must be dict or list of spec parameters')
            specs.append(spec)
        priors = generate_ssd_priors(specs, image_size)
        return cls(image_size=image_size,
                   image_mean=image_mean,
                   image_std=json_dict['image_std'],
                   iou_threshold=json_dict['iou_threshold'],
                   center_variance=json_dict['center_variance'],
                   size_variance=json_dict['size_variance'],
                   specs=specs,
                   priors=priors)

    @classmethod
    def from_json(cls, json_path):
        json_dict = json.load(open(json_path, 'r'))
        return cls.from_json_dict(json_dict)
