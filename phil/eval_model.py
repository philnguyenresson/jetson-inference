
import os
import sys
import logging
import argparse
import itertools
import torch
import inspect
from os.path import join as pjoin
import numpy as np

#relative path hack
test_path = os.path.join(os.path.dirname(__file__), '..', 'Common')
test = os.path.join(sys.path[0], '..')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
python_ssd_traindir = os.path.join(parentdir,'python','training','detection','ssd')
sys.path.append(python_ssd_traindir)

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

import cv2

def main(args):
    net_type = args.net_type

    model_path = args.model_path
    label_path = args.label_path
    image_folder = args.image_folder
    output_folder = args.output_folder

    class_names = [name.strip() for name in open(label_path).readlines()]
    DEVICE = torch.device("cuda:0")

    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200,device=DEVICE)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200,device=DEVICE)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200,device=DEVICE)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=DEVICE)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200,device=DEVICE)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200,device=DEVICE)

    os.makedirs(output_folder,exist_ok=True)

    for file in os.listdir(image_folder):
        if os.path.splitext(file)[-1] != '.jpg':
            continue
        image_path = pjoin(image_folder,file)
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        # image = image.to(DEVICE)
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        if boxes.size(0) == 0:
            print('no anom found, skipping')
            continue
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        
        cv2.imwrite(pjoin(output_folder,file), orig_image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='')

    # Params for datasets
    parser.add_argument("--net_type", required=True, 
                        dest='net_type',
                        type=str,
                        help='')
    parser.add_argument("--model_path", required=True, 
                        dest='model_path',
                        type=str,
                        help='')
    parser.add_argument("--label_path", required=True, 
                        dest='label_path',
                        type=str,
                        help='')
    parser.add_argument("--image_folder", required=True, 
                        dest='image_folder',
                        type=str,
                        help='')
    parser.add_argument("--output_folder", required=True, 
                        dest='output_folder',
                        type=str,
                        help='')


    args = parser.parse_args()
    main(args)