from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf
import glob
import time

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

flags.DEFINE_string('cfg_path', './configs/train.yaml', 'config file path')
flags.DEFINE_string('gpu', '1', 'which gpu to use') # -1 means using cpu
flags.DEFINE_string('down_ckpt_dir','./', 'where to load trained model')
flags.DEFINE_string('ckpt_dir','./checkpoints', 'where to load trained model')

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from train_DRRN import *

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

def main(_argv):
    need_h5 = 0
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    cfg = load_yaml(FLAGS.cfg_path)
    # define network
    model = DRRNet(None, cfg['ch_size'])
    # load checkpoint
    # checkpoint_dir = FLAGS.down_ckpt_dir # use downloaded ckpt
    checkpoint_dir = FLAGS.ckpt_dir # use trained ckpt
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_name = ""
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] loaded ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
        ckpt_name = get_image_name(str(tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    h5_path = "./h5_files/DRRN_"+str(ckpt_name+".h5")
    if(need_h5==1):
        model.save(h5_path) # need to rewrite customized layer's get_config() function to save model
        print("h5 file generated! ")
    print("Start converting... ")
    pb_path = "./pb_files"
    os.system("mkdir "+pb_path)
    pb_name = "DRRN_"+str(ckpt_name+".pb")

    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)
    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=pb_path,
                      name=pb_name,
                      as_text=False)
    print("pb file generated! ")


if __name__ == '__main__':
    app.run(main)


