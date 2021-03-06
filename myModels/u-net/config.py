
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""

C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]


dir_str =  C.abs_dir
lst =dir_str.split('/')
lst.pop(-1)
lst.pop(-1)
delima = '/'
res = delima.join(lst)
print(res)
C.root_dir = res
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = osp.abspath(osp.join(C.root_dir, 'data'))
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "val.txt")
C.test_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'core'))


"""Image Config"""
C.num_classes = 2
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.target_size = 1024
C.image_height = 400
C.image_width = 400
C.test_image_height = 608
C.test_image_width = 608
C.gt_down_sampling = 1
C.num_train_imgs = 90
C.num_eval_imgs = 10

""" Settings for network, this would be different for each kind of myModels"""
C.fix_bias = True
C.fix_bn = False
C.sync_bn = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1
C.pretrained_model = None

"""Train Config"""
C.lr = 1e-3
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 5e-4
C.batch_size = 2
C.nepochs = 40
C.niters_per_epoch = 1000
C.num_workers = 4
C.train_scale_array = [0.75, 1, 1.25, 1.5, 1.75, 2.0]

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]
C.eval_flip = False
C.eval_height = 400
C.eval_width = 400

"""Display Config"""
C.snapshot_iter = 50
C.record_info_iter = 20
C.display_iter = 50

if __name__ == '__main__':
    print("epoch_num: ", C.nepochs)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

