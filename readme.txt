env build: build conda and pip envs in conda_env folder accordingly
train: python train_DRRN.py
test PSNR,SSIM: python test_DRRN.py
make pb file: python make_pb.py

warning: infering can only be done by CPU since GPU mem is too small and parameters are numerous