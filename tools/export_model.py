# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np

import os
import random
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from src.models.builder import build_classifier
from src.utils import build_optim, build_lrscheduler, LRSchedulerC, VisualDLC, build_transform
from tools.configsys import CfgNode as CN

from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import Cifar10

paddle.set_device("cpu")

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


def get_args(add_help=True):
    """get_args

    Parse all args using argparse lib

    Args:
        add_help: Whether to add -h option on args

    Returns:
        An object which contains many parameters used for inference.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--img-size', default=32, help='image size to export')
    parser.add_argument(
        '--save-inference-dir', default='.', help='path where to save')
    parser.add_argument('--pretrained', default='logs/resnet50-supcon-2022-03-09-06:48:11/0.pdparams', help='pretrained model')
    parser.add_argument('--num-classes', default=1000, help='num_classes')
    parser.add_argument('-y', '--yaml', default='config/resnet50_linear.yml', type=str)
    parser.add_argument('--test', action='store_true', default=True, help='test only')
    args = parser.parse_args()
    return args


def export(args):
    # build model
    cfg = CN.load_cfg(args.yaml)
    cfg.COMMON.test_only = args.test
    cfg.freeze()
    paddle.seed(cfg.COMMON.seed)
    random.seed(cfg.COMMON.seed)
    np.random.seed(cfg.COMMON.seed)

    model = build_classifier(cfg.CLASSIFIER)
    # print(model)

    state_dict = paddle.load(args.pretrained)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("encoder.", "")
        new_state_dict[k] = v
    model.set_state_dict(new_state_dict)
    model = nn.Sequential(model, nn.Softmax(axis=-1))
    model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
