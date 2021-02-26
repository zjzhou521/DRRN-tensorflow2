from train_DRRN import *

def test(is_validate):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("using cpu to infer")

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    # set_memory_growth()
    cfg = load_yaml('./configs/train.yaml')
    # define network (Generator)
    model = DRRNet(None, cfg['ch_size'])
    # model.summary(line_length=110)
    # load checkpoint
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] loaded ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    if(is_validate!=1): # only infer high scale
        inputs_path = "./data/Set14"
        outputs_path = "./data/Set14_4x"
        if(not os.path.isdir(outputs_path)):
            os.makedirs(outputs_path)
            print("[*] created output path = ",outputs_path)
        print("[*] %s -> %s"%(inputs_path,outputs_path))
        print("[*] inferring, scale = 4")
        image_path_list = get_image_path(inputs_path)
        total_num = len(image_path_list)
        cnt = 0
        for img_path in image_path_list:
            t_start = time.time()
            cnt += 1
            img_name = get_image_name(img_path)
            print("[*] processing[%d/%d]:%s"%(cnt,total_num,img_name))
            raw_img = cv2.imread(img_path)
            print("   [*] raw image shape = ",raw_img.shape)
            lr_img = raw_img
            sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_ESRGAN_4x" + img_name_after
            output_img_path = os.path.join(outputs_path, output_img_name)
            outputs_img = sr_img
            print("output_img_name = ",output_img_name)
            cv2.imwrite(output_img_path, outputs_img)
            t_end = time.time()
            print("   [*] done! Time = %.1fs"%(t_end - t_start))
    else: # generate low res and compare with metrics
        inputs_path = "./data/Set14"
        outputs_path = "./data/Set14_4x"
        if(not os.path.isdir(outputs_path)):
            os.makedirs(outputs_path)
            print("[*] created output path = ",outputs_path)
        print("[*] %s -> %s"%(inputs_path,outputs_path))

        print("   image_name                   PSNR/SSIM        PSNR/SSIM (higher,better)")
        image_path_list = get_image_path(inputs_path)
        for img_path in image_path_list:
            img_name = get_image_name(img_path)
            raw_img = cv2.imread(img_path)
            # Generate low resolution image with original images
            lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale']) # scale=4
            # lr_img = raw_img
            model_input = lr_img[np.newaxis, :] / 255
            model_input = model_input.astype('float32')
            # print("model_input.dtype = ",model_input.dtype) # 'float32' instead of float64
            # print("model_input shape = ",model_input.shape)
            model_output = model(model_input)
            sr_img = tensor2img(model_output)
            bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)
            str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
            print(str_format.format(
                img_name + ' ' * max(0, 20 - len(img_name)),
                calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_ESRGAN_025x_4x" + img_name_after
            output_img_path = os.path.join(outputs_path, output_img_name)
            # outputs_img = np.concatenate((bic_img, sr_img, hr_img), 1)
            outputs_img = sr_img
            # cv2.imwrite(output_img_path, outputs_img)
            img_name_before, img_name_after = divide_image_name(img_name)
            output_img_name = img_name_before + "_ESRGAN_025x" + img_name_after
            output_lr_img_path = os.path.join(outputs_path, output_img_name)
            outputs_lr_img = lr_img
            # cv2.imwrite(output_lr_img_path, outputs_lr_img) # write low resoltion images
    print("[*] Done!")


if __name__ == '__main__':
    is_validate = 1
    test(is_validate)















