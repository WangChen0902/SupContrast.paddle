import os
import random
import time
import numpy as np
import argparse
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from src.models.builder import build_classifier
from src.utils import build_optim, build_lrscheduler, LRSchedulerC, VisualDLC, build_transform
from tools.configsys import CfgNode as CN

import paddle
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import Cifar10
from PIL import Image
import paddle.vision.transforms as transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Set PaddlePaddle cifar10 config', add_help=False)
    parser.add_argument('-y', '--yaml', default='config/resnet50_linear.yml', type=str)
    parser.add_argument('--img-path', default='./images/supcon_test.png', type=str)
    parser.add_argument('--test', action='store_true', default=True, help='test only')
    return parser


def main(cfg, img_path):
    paddle.seed(cfg.COMMON.seed)
    random.seed(cfg.COMMON.seed)
    np.random.seed(cfg.COMMON.seed)

    net = build_classifier(cfg.CLASSIFIER)
    # print(net)

    model = paddle.Model(net)

    lrs = build_lrscheduler(cfg.SCHEDULER)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1)
    optim = build_optim(cfg.OPTIMIZER, parameters=net.parameters(), learning_rate=lrs, grad_clip=clip)
    train_transforms, val_transforms = build_transform()
    # train_set = Cifar10(cfg.COMMON.data_path, mode='train', transform=train_transforms)
    # test_set = Cifar10(cfg.COMMON.data_path, mode='test', transform=val_transforms)
    vis_name = '/{}-{}-{}'.format(cfg.CLASSIFIER.name, cfg.CLASSIFIER.mode,
                                  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    log_dir = cfg.COMMON.logdir + vis_name
    callbacks = [LRSchedulerC(), VisualDLC(log_dir)]

    model.prepare(optim, CrossEntropyLoss(), Accuracy(topk=(1, 5)))

    # load SupCon encoder
    # to freeze the encoder weights and only train the classifier, set "only_fc: True" in config file.
    if 'from_supcon' in cfg.COMMON:
        print("Loading SupCon encoder state from", cfg.COMMON.from_supcon)
        state_dict = paddle.load(cfg.COMMON.from_supcon + ".pdparams")
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("encoder.", "")
            new_state_dict[k] = v
        model.network.set_state_dict(new_state_dict)

    # load checkpoint
    if 'continue_from' in cfg.COMMON:
        model.load(cfg.COMMON.continue_from)

    if cfg.COMMON.test_only:
        # label, data = next(enumerate(test_set))
        train_transforms, val_transforms = build_transform()
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        # img = transforms.to_tensor(img)
        img = transforms.Resize(size=(32, 32))(img)
        img = val_transforms(img).unsqueeze(0)
        print(img.shape)
        print(img.mean())
        # img = np.expand_dims(img, axis=0)
        # data_tensor, data_array = data
        predict = paddle.to_tensor(model.predict_batch(img))
        print(predict)
        class_id = predict[0].argmax()
        predict = paddle.nn.functional.softmax(predict[0])
        prob = predict[0][0][class_id]
        print("class_id: {}, prob: {}".format(class_id, prob))
    else:
        model.fit(
            train_set,
            test_set,
            batch_size=cfg.COMMON.batch_size,
            epochs=cfg.COMMON.epochs,
            num_workers=cfg.COMMON.workers,
            save_dir=log_dir,
            save_freq=cfg.COMMON.save_freq,
            verbose=cfg.COMMON.verbose,
            callbacks=callbacks,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PaddlePaddle cifar10 classifier training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = CN.load_cfg(args.yaml)
    cfg.COMMON.test_only = args.test
    cfg.freeze()
    print(cfg)
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    print("num of GPUs:", n_gpu)
    if n_gpu > 1:
        paddle.distributed.spawn(main, args=(cfg, args.img_path), nprocs=n_gpu)
    else:
        main(cfg, args.img_path)
