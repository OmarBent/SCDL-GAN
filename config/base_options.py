import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # Change for each experiment
        file_name = "Florence"
        self.parser.add_argument('--dataroot', type=str, default='Data/' + file_name,
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--rawdatapath', type=str, default='raw_training_videos/Weizmann',
                                 help='path to raw training videos directory')
        self.parser.add_argument("--output_dir", type=str, default='Data/' + file_name + '/Pose-GAN',
                                 help="save directory")
        self.parser.add_argument("--file_to_save", type=str, default='Data/' + file_name + '/SC-DL',
                                 help="file to save SCDL outputs")
        self.parser.add_argument('--name', type=str, default=file_name,
                                 help='name of the experiment. It decides where to store samples and models')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./PT_checkpoints', help='models are saved here')
        self.parser.add_argument("--seq_length", type=int, default=30, help="sequences are re-sampled to same length")
        self.parser.add_argument("--n_gen", type=int, default=1000, help="number of sequences to generate at last epoch")
        self.parser.add_argument("--image_width", type=int, default=180, help="image width")
        self.parser.add_argument("--image_height", type=int, default=144, help="image height")
        self.parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")
        self.parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")

        # Common parameters
        self.parser.add_argument('--openpose_dir', type=str, default='C:/Users/BENTANFOUS/Desktop/Pose-Transfer/openpose/bin', help='openpose command directory')
        self.parser.add_argument('--openface_dir', type=str, default='C:/Users/BENTANFOUS/Desktop/shapeVGAN/OpenFace_2.2.0_win_x64', help='openface command directory')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='resnet', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='PATN', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='blocks used in D')
        self.parser.add_argument('--dataset_mode', type=str, default='keypoint', help='choose how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='PATN',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='no', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--P_input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--BP_input_nc', type=int, default=18, help='# of input image channels')
        self.parser.add_argument('--padding_type', type=str, default='reflect', help='# of input image channels')


        self.parser.add_argument('--with_D_PP', type=int, default=1, help='use D to judge P and P is pair or not')
        self.parser.add_argument('--with_D_PB', type=int, default=1, help='use D to judge P and B is pair or not')

        self.parser.add_argument('--use_flip', type=int, default=0, help='flip or not')

        # down-sampling times
        self.parser.add_argument('--G_n_downsampling', type=int, default=2, help='down-sampling blocks for generator')
        self.parser.add_argument('--D_n_downsampling', type=int, default=2, help='down-sampling blocks for discriminator')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

