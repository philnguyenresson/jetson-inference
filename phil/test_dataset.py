import os
import sys
import logging
import argparse
import itertools
import torch
import inspect

#relative path hack
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
python_ssd_traindir = os.path.join(parentdir,'python','training','detection','ssd')
sys.path.append(python_ssd_traindir)


from vision.datasets.open_images import OpenImagesDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='')
    

    # Params for datasets
    parser.add_argument("--dataset_path", required=True, 
                        dest='dataset_path',
                        type=str,
                        help='')
    args = parser.parse_args()

    create_net = create_mobilenetv1_ssd
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    val_dataset = OpenImagesDataset(args.dataset_path,
                                    transform=test_transform, target_transform=target_transform,
                                    dataset_type="test")
    