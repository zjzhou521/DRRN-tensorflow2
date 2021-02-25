# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ReLU, Add, concatenate, Input, Conv2D, Conv2DTranspose
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from modules.models import RRDB_Model
from modules.lr_scheduler import MultiStepLR
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)
import glob
import cv2
import time
import numpy as np
from modules.dataset import load_tfrecord_dataset


def get_image_path(dir_name):
    pattern_name = dir_name + '/**/*.[jbptJBPT][pnmiPNMI][gepfGEPF]'
    image_paths=[]
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    pattern_name = dir_name + '/**/*.[jtJT][piPI][efEF][gfGF]'
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    return image_paths
def get_image_name(path):
    i = 0
    name_flag = 0
    image_name = ""
    if(path[0]=='.'): i += 2
    while(i<len(path)):
        if(name_flag!=0):
            image_name += path[i]
        if(path[i]=='/'):
            name_flag = 1
        i += 1
    if('/' in image_name):
        return get_image_name(image_name)
    else:
        return image_name
def divide_image_name(name):
    i = 0
    before = ""
    after = ""
    divide_flag = 0
    while(i<len(name)):
        if(name[i]=='.'):
            divide_flag = 1
        if(divide_flag==0):
            before += name[i]
        else:
            after += name[i]
        i += 1
    return before, after
def load_dataset(cfg, key, shuffle=True, buffer_size=10240):
    """load dataset"""
    dataset_cfg = cfg[key]
    logging.info("load {} from {}".format(key, dataset_cfg['path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=dataset_cfg['path'],
        batch_size=cfg['batch_size'],
        gt_size=cfg['gt_size'],
        scale=cfg['scale'],
        shuffle=shuffle,
        using_bin=dataset_cfg['using_bin'],
        using_flip=dataset_cfg['using_flip'],
        using_rot=dataset_cfg['using_rot'],
        buffer_size=buffer_size)
    return dataset


def DRRNet(size, channels, B=25, name="DRRNet"):
    x = ori_input = Input([size, size, channels], name='input_image')
    # upsample with bicubic
    size_x_h = tf.shape(x)[1] if size is None else size
    size_x_w = tf.shape(x)[2] if size is None else size
    # Nearest Method (simple linear interpolation)
    x = tf.image.resize(x, [size_x_h * 4, size_x_w * 4],method='nearest', name='upsample')
    residual = x
    inputs = ReLU()(x)
    inputs = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None,use_bias=False, name="conv2d_input")(inputs)
    out = inputs
    for i in range(25):
        out = ReLU()(out)
        out = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None, use_bias=False, name="conv2d_B%d_1" % (i+1))(out)
        out = ReLU()(out)
        out = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None, use_bias=False, name="conv2d_B%d_2" % (i+1))(out)
        out = Add()([out,inputs])
    out = ReLU()(out)
    out = Conv2D(filters=channels, kernel_size=3, strides=1, padding="same", activation=None, use_bias=False, name="conv2d_output")(out)
    out = Add()([out,residual])
    return Model(ori_input, out, name=name)

def train():
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml('./configs/train.yaml')

    # define network (Generator)
    model = DRRNet(cfg['input_size'], cfg['ch_size'])
    model.summary(line_length=110)
    plot_model(model, to_file='model_vis.png',show_shapes=True) # plot model
    # load dataset with shuffle
    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=True)

    # define Adam optimizer
    learning_rate = MultiStepLR(cfg['lr'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=cfg['adam_beta1_G'],
                                         beta_2=cfg['adam_beta2_G'])

    # define loss function
    mean_squared_loss_fn = tf.keras.losses.MeanSquaredError()

    # load checkpoint
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=50)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        # get output and loss
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)
            total_loss = mean_squared_loss_fn(hr, sr)
        # optimizer
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    summary_writer = tf.summary.create_file_writer('./logs')
    # prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)
    cnter = remain_steps
    # start training
    for lr, hr in train_dataset.take(remain_steps):
        cnter -= 1
        t_start = time.time()
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        total_loss = train_step(lr, hr)
        # visualize
        # prog_bar.update("loss={:.4f}, lr={:.1e}".format(total_loss.numpy(), optimizer.lr(steps).numpy()))
        stps_epoch = int(cfg['train_dataset']['num_samples']/cfg['batch_size'])
        t_end = time.time()
        # print information
        print_steps = 2013
        if(steps%print_steps==0):
            print("epoch=%3d step=%4d/%d loss=%3.4f lr=%.5f stp_time=%.3f cnter=%6d"%(int(steps/stps_epoch),int(steps%stps_epoch),stps_epoch,total_loss.numpy(),optimizer.lr(steps).numpy(),t_end-t_start,cnter))
        # log loss and leanring rate
        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)
        # save checkpoint
        save_epoch = 5
        if(steps % stps_epoch == 0 and (steps/stps_epoch)%save_epoch==0): # each ckpt = 5 epoches
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))
    print("\n[*] training done!")


if __name__ == '__main__':
    train()















