import configargparse
import copy
import numpy as np
import time
import os
import errno
import os.path as osp
import subprocess
import random
import cv2 as cv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

import utils_func
from utils.logger import set_logger, get_logger
from utils.avg_meters import AverageMeter
from data.kitti_loader_lidar import KittiDataset_Fusion_stereo
from kitti_evaluate import predict_kitti_to_file
from kitti_process_detection import gen_feature_diffused_tensor, get_3D_global_grid_extended
from network.pixor_fusion import PixorNet_Fusion

from monodepth.layers import * #disp_to_depth, transformation_from_parameters, BackprojectDepth, Project3D
from monodepth.utils import readlines
from monodepth.options import MonodepthOptions
from monodepth import datasets
from monodepth import networks
from torchvision.utils import save_image

MONO_SCALE_FACTOR = 31.286
MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def get_grid_2D(zsize, xsize):
    z = np.linspace(0.0 + 70.0 / zsize / 2, 70.0 - 70.0 / zsize / 2, num=zsize)
    x = np.linspace(-40.0 + 80.0 / xsize / 2, 40.0 -
                    80.0 / xsize / 2, num=xsize)
    pc_grid = np.zeros((2, zsize, xsize))
    for i in range(zsize):
        for j in range(xsize):
            pc_grid[:, i, j] = [x[j], z[i]]
    return pc_grid


label_grid = get_grid_2D(175, 200)


def parse_args():
    parser = configargparse.ArgParser(
        description="Train PIXOR model")
    parser.add('-c', '--config', required=True,
        is_config_file=True, help='config file')
    parser.add_argument("--mode",
        choices=["train", "eval"], type=str, default="train")
    parser.add_argument("--eval_dataset",
        choices=["train", "val"], type=str, default="val")
    parser.add_argument('--train_dataset', default='train',
        choices=["train"])

    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument('--image_sets', default='./image_sets')

    parser.add_argument('--root_dir', type=str,
        default="/scratch/datasets/KITTI/object")
    parser.add_argument('--raw_dir', type=str,
        default="/root/dataset/kitti_raw1")    
    parser.add_argument('--train_label_dir', type=str, default='label_2')
    parser.add_argument('--eval_label_dir', type=str, default='label_2')
    parser.add_argument('--train_calib_dir', type=str, default='calib')
    parser.add_argument('--eval_calib_dir', type=str, default='calib')
    parser.add_argument("--train_lidar_dir", type=str, default='velodyne')
    parser.add_argument("--eval_lidar_dir", type=str, default='velodyne')
    parser.add_argument('--train_image_dir', type=str, default='image_2')
    parser.add_argument('--eval_image_dir', type=str, default='image_2')
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--e2e', action='store_true')

    # depth_model
    parser.add_argument("--depth_down", type=int, default=2)
    parser.add_argument("--depth_lr", type=float, default=1e-3)
    parser.add_argument('--depth_lr_stepsize', nargs='+', type=int,
                        default=[10],
                        help='learning rate decay step size')
    parser.add_argument('--depth_lr_gamma', default=0.1, type=float,
                        help='gamma for learning rate decay')
    parser.add_argument('--depth_pretrain', default=None,
                        help='load model')
    parser.add_argument("--diffused", action="store_true")
    parser.add_argument("--depth_loss", type=str,choices=["M", "MD", "D"],default="D")

    # hyperparameters
    parser.add_argument('--pixor_pretrain', default=None,
                        help='load model')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--flip_rate", type=float, default=0.5)
    parser.add_argument("--random_shift_scale", type=float, default=5)
    parser.add_argument("--groupnorm", action="store_true")

    parser.add_argument("--pixor_fusion", action='store_true')
    parser.add_argument("--resnet_type", type=str, default='resnet50')
    parser.add_argument("--image_downscale", type=int, default=1)
    parser.add_argument("--resnet_chls", type=int, default=64,
                        help='output channels')
    parser.add_argument("--no_reflex", action="store_true")
    parser.add_argument("--crop_height", type=int, default=-1)

    # optimizer
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opt_method", default='RMSprop')
    parser.add_argument("--lr_milestones", nargs='+',
                        help='lr decay schedule at specified epoches by 0.1',
                        type=int, default=[11])
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # save model
    parser.add_argument("--datapath", type=str,
                        default="/scratch/datasets/KITTI")
    parser.add_argument("--saverootpath", type=str,
                        default="/scratch/HaLF/save")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--jobid", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth.tar")
    parser.add_argument("--save_every", type=int, default=1)

    # evaluate
    parser.add_argument("--eval_save_prefix", type=str, default="")
    parser.add_argument("--start_eval", type=int, default=10)
    parser.add_argument("--gen_predict_file", action="store_true")
    parser.add_argument("--no_cal_loss", action="store_true")
    parser.add_argument("--gen_slow", action="store_true")
    parser.add_argument("--run_official_evaluate", action="store_true")
    parser.add_argument("--evaluate_bin7", type=str,
        default="./evaluation/kitti_eval/evaluate_object_3d_offline07")
    parser.add_argument("--evaluate_bin5", type=str,
        default="./evaluation/kitti_eval/evaluate_object_3d_offline05")
    parser.add_argument("--throw_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.1)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--eval_ckpt", type=str, default=None)

    # logging
    parser.add_argument("--shortname", help="name displayed on logging",
                        type=str, default=None)
    parser.add_argument("--loglevel", help="logging levels",
                        type=str, default="INFO")
    parser.add_argument("--logevery", help="log every x batches",
                        type=int, default=100)

    parser.add_argument("--show_eval_progress", action="store_true")

    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) == 'None':
            setattr(args, arg, None)

    return args


def compute_loss(epoch, class_outs, reg_outs,
                 class_labels, reg_labels,
                 class_criterion, reg_criterion, args):

    class_loss = (class_criterion(class_outs, class_labels)).mean()

    reg_loss = 0
    count = 0
    for i in range(args.batch_size):
        valid_pos = class_labels[i:i + 1] > 0
        if int(valid_pos.sum().data) == 0:
            continue
        reg_loss += (reg_criterion(reg_outs[i:i + 1],
                                   reg_labels[i:i + 1]).permute(
                                        0, 2, 3, 1))[valid_pos].sum()
        count += valid_pos.sum().float()
    if count > 0:
        reg_loss /= count
    loss = class_loss + reg_loss
    return class_loss, reg_loss, loss


def forward(epoch, model, inputs, class_labels, reg_labels,
            class_criterion, reg_criterion, args, get_intermediate=False):

    class_outs, reg_outs = model(x=inputs)
    class_outs = class_outs.squeeze(1)
    return compute_loss(epoch, class_outs, reg_outs,
                        class_labels, reg_labels,
                        class_criterion, reg_criterion, args)


def display_args(args, logger):
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    logger.info("========== training info ==========")
    logger.info("host: {}".format(os.getenv('HOSTNAME')))
    logger.info("gpu: {}".format(use_gpu))
    if use_gpu:
        logger.info("gpu_dev: {}".format(num_gpu))

    for arg in vars(args):
        logger.info("{} = {}".format(arg, getattr(args, arg)))
    logger.info("===================================")


def get_eval_dataset(args):
    eval_file = os.path.join(
        args.image_sets, "{}.txt".format(args.eval_dataset))
    n_features = 35 if args.no_reflex else 36
    eval_data = KittiDataset_Fusion_stereo(txt_file=eval_file,
                                    flip_rate=0,
                                    random_shift_scale=0,
                                    lidar_dir=args.eval_lidar_dir,
                                    label_dir=args.eval_label_dir,
                                    calib_dir=args.eval_calib_dir,
                                    image_dir=args.eval_image_dir,
                                    root_dir=args.root_dir,
                                    raw_dir = args.raw_dir,# crkim "/root/dataset/kitti_raw"
                                    only_feature=args.no_cal_loss,
                                    split=args.split,
                                    image_downscale=args.image_downscale,
                                    crop_height=args.crop_height)
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return eval_data, eval_loader


def forward_monodepth_model(imgL, color, depth, metric_log, depth_model, encoder_model, mode='TRAIN'):
    # ---------
    mask = (depth >= 1) * (depth <= 80)
    mask.detach_()
    min_depth = 0.1  # while training
    max_depth = 100
    # ----
    disp = depth_model(encoder_model(color))[-1]
    # pred_disp, _ = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
    pred_disp, _ = disp_to_depth(disp, min_depth, max_depth)
    pred_disp = F.interpolate(pred_disp, size=(imgL.shape[2], imgL.shape[3]), mode="bilinear", align_corners=False).squeeze(1)
    pred_depth = 1 / pred_disp
    pred_depth *= MONO_SCALE_FACTOR
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    loss = F.smooth_l1_loss(pred_depth[mask], depth[mask], size_average=True)
    metric_log.calculate(depth, pred_depth, loss=loss.item())
    return loss, pred_depth, disp, mask

def compute_reprojection_loss(target, pred):
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim = SSIM().cuda()
    ssim_loss = ssim(target, pred).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return reprojection_loss


def forward_pose_model(args, intrinsic, pose_encoder, pose_decoder, norm_image_seq, image_seq, disp, iteration, mask):
    """Predict poses between input frames for monocular sequences.
    """
    color = norm_image_seq['color']# b*1*192*640
    color_post = norm_image_seq['color_post']
    color_prev = norm_image_seq['color_prev']
    reprojection_losses = []
    loss = 0
    for idx, pose_input in enumerate([[color_prev, color], [color, color_post]]):
        pose_input = [pose_encoder(torch.cat(pose_input, 1))]
        axisangle, translation = pose_decoder(pose_input)
        cam_T_cam = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(idx == 0)) #invert=(idx == 0))

        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        min_depth = 0.1  # while training
        max_depth = 100
        _, depth = disp_to_depth(disp, min_depth, max_depth) # b*1*192*640
        depth = F.interpolate(depth, size=(mask.shape[1], mask.shape[2]), mode="bilinear", align_corners=False)
        #depth *= MONO_SCALE_FACTOR
        #depth[depth < MIN_DEPTH] = MIN_DEPTH
        #depth[depth > MAX_DEPTH] = MAX_DEPTH

        K = intrinsic.float()
        inv_K = torch.inverse(K)

        _, h, w = mask.shape 

        backproject_depth = BackprojectDepth(inv_K.shape[0], h, w).cuda()
        project_3d = Project3D(inv_K.shape[0], h, w).cuda()
        cam_points = backproject_depth(depth, inv_K)
        pix_coords = project_3d(cam_points, K, cam_T_cam[:,:3,:])
        curr_image =  image_seq['curr_image'] # b*3*H_ori *W_ori
        if idx ==0 :
            adjacent_img = image_seq['prev_image']
        else :
            adjacent_img = image_seq['post_image']
        warped_img = F.grid_sample(adjacent_img, pix_coords, padding_mode="border", align_corners=False)
        """Computes reprojection loss between a batch of predicted and target images
        """
        reprojection_loss = compute_reprojection_loss(curr_image, warped_img)
        reprojection_losses.append(reprojection_loss)

        
        if idx == 0:
            prev_image = adjacent_img
            prev_warp_img = warped_img
        else:
            post_image = adjacent_img
            post_warp_img = warped_img

    if iteration % 100 == 0:
        #save
        save_depth = depth
        savepath = osp.join(args.saverootpath, args.run_name)
        if not osp.exists(savepath):
            os.makedirs(savepath)
        for i in range(inv_K.shape[0]):
            save_depth[i] = depth[i]/ depth[i].max()
        save_image(save_depth, osp.join(savepath, "depth.png"))
        colors = torch.cat((color_prev, color, color_post), dim=0)
        save_image(colors, osp.join(savepath, "colors_norm.png"),nrow= 6)
        images = torch.cat((prev_image, curr_image, post_image), dim=0)    
        save_image(images, osp.join(savepath, "images.png"),nrow= 6)
        warps = torch.cat((prev_warp_img, curr_image, post_warp_img), dim=0)    
        save_image(warps, osp.join(savepath, "warps.png"),nrow= 6)
        save_image(mask.unsqueeze(1).int()*1.1, osp.join(savepath, "masks.png"),nrow= 6)

    #return (reprojection_losses[0].mean()+reprojection_losses[1].mean())/2
    
    reprojection_losses = torch.cat(reprojection_losses, 1)
    identity_reprojection_losses = []
    for img_seq in ([[image_seq['prev_image'], image_seq['curr_image']], [image_seq['curr_image'], image_seq['post_image']]]):
        identity_reprojection_loss = compute_reprojection_loss(img_seq[0], img_seq[1])
        identity_reprojection_losses.append(identity_reprojection_loss)
    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
    identity_reprojection_loss = identity_reprojection_losses
    reprojection_loss = reprojection_losses
    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
    to_optimise, idxs = torch.min(combined, dim=1)    #min loss
    if args.depth_loss == "MD" :
        loss += to_optimise[~mask].mean()
    else: 
        loss += to_optimise.mean()
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    smooth_loss = get_smooth_loss(norm_disp, color)
    loss += 1e-3* smooth_loss
    return loss
    

def depth_to_pcl(calib, depth, max_high=1.):
    rows, cols = depth.shape
    c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
    points = torch.stack([c.t(), r.t(), depth], dim=0)
    points = points.reshape((3, -1))
    points = points.t()
    cloud = calib.img_to_lidar(points[:, 0], points[:, 1], points[:, 2])
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    lidar = cloud[valid]

    # pad 1 in the intensity dimension
    lidar = torch.cat(
        [lidar, torch.ones((lidar.shape[0], 1), device='cuda')], 1)
    lidar = lidar.float()
    return lidar


def train(args):
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    assert use_gpu, "Please use gpus."

    logger = get_logger(name=args.shortname)
    display_args(args, logger)

    # create dir for saving
    args.saverootpath = osp.abspath(args.saverootpath)
    savepath = osp.join(args.saverootpath, args.run_name)
    if not osp.exists(savepath):
        os.makedirs(savepath)

    train_file = os.path.join(
        args.image_sets, "{}.txt".format(args.train_dataset))
    n_features = 35 if args.no_reflex else 36
    train_data = KittiDataset_Fusion_stereo(
        txt_file=train_file,
        flip_rate=args.flip_rate,
        lidar_dir=args.eval_lidar_dir,
        label_dir=args.eval_label_dir,
        calib_dir=args.eval_calib_dir,
        image_dir=args.eval_image_dir,
        root_dir=args.root_dir,
         raw_dir = args.raw_dir, #crkim
        only_feature=args.no_cal_loss,
        split=args.split,
        image_downscale=args.image_downscale,
        crop_height=args.crop_height,
        random_shift_scale=args.random_shift_scale)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)

    eval_data, eval_loader = get_eval_dataset(args)

    pixor = PixorNet_Fusion(n_features, groupnorm=args.groupnorm,
                            resnet_type=args.resnet_type,
                            image_downscale=args.image_downscale,
                            resnet_chls=args.resnet_chls)

    ts = time.time()
    pixor = pixor.cuda()  # to(torch.device("cuda"))
    pixor = nn.DataParallel(pixor, device_ids=num_gpu)

    class_criterion = nn.BCELoss(reduction='none')
    reg_criterion = nn.SmoothL1Loss(reduction='none')

    if args.opt_method == 'RMSprop':
        optimizer = optim.RMSprop(pixor.parameters(), lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    grid_3D_extended = get_3D_global_grid_extended(700, 800, 35).cuda().float()

    #--options---
    num_layers = 18
    depth_lr = 1e-4
    scheduler_step_size = 15
    num_pose_frames = 2
    weights_init = "pretrained"
    #frame_ids = [0, 1]
    # -----------
    parameters_to_train = []
    encoder = networks.ResnetEncoder(num_layers, False).cuda() # to(torch.device("cuda"))
    encoder = nn.DataParallel(encoder, device_ids=num_gpu)
    parameters_to_train += list(encoder.parameters())
    depth_decoder = networks.DepthDecoder(encoder.module.num_ch_enc).cuda()  # .to(torch.device("cuda"))
    depth_decoder = nn.DataParallel(depth_decoder, device_ids=num_gpu)
    parameters_to_train += list(depth_decoder.parameters())
    #depth_optimizer = optim.Adam(parameters_to_train, lr=depth_lr)
    if "M" in args.depth_loss:
        pose_encoder = networks.ResnetEncoder(num_layers, weights_init, num_input_images=num_pose_frames).cuda()
        pose_encoder = nn.DataParallel(pose_encoder, device_ids=num_gpu)
        parameters_to_train += list(pose_encoder.parameters())
        
        pose_decoder = networks.PoseDecoder(pose_encoder.module.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).cuda()   
        pose_decoder = nn.DataParallel(pose_decoder, device_ids=num_gpu)            
        parameters_to_train += list(pose_decoder.parameters())

    depth_optimizer = optim.Adam(parameters_to_train, lr=depth_lr)

    if args.depth_pretrain:
        encoder_path = os.path.join(args.depth_pretrain, "encoder.pth")
        decoder_path = os.path.join(args.depth_pretrain, "depth.pth")
        if os.path.isfile(encoder_path) and os.path.isfile(decoder_path):
            logger.info("=> loading depth pretrain '{}'".format(args.depth_pretrain))
            encoder_dict = torch.load(encoder_path)
            encoder_dict = {'module.'+k: v for k, v in encoder_dict.items() if k not in ['height', 'width', 'use_stereo']}
            encoder.load_state_dict(encoder_dict)
            decoder_dict = torch.load(decoder_path)
            #decoder_dict = {'module.'+k: v for k, v in decoder_dict.items()}
            #decoder_dict = {k: v for k, v in decoder_dict.items()}
            #depth_decoder.load_state_dict(decoder_dict, strict=False)]
            key_list = list(decoder_dict.keys())
            new_key_list = list(depth_decoder.state_dict().keys())
            new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
            decoder_dict = {new_key_map[k]: v for k, v in decoder_dict.items()}
            depth_decoder.load_state_dict(decoder_dict)
            if "M" in args.depth_loss:
                pose_encoder_path = os.path.join(args.depth_pretrain, "pose_encoder.pth")
                pose_decoder_path = os.path.join(args.depth_pretrain, "pose.pth")
                if os.path.isfile(pose_encoder_path) and os.path.isfile(pose_decoder_path):
                    logger.info("=> loading pose encoder pretrain '{}'".format(pose_encoder_path))
                    logger.info("=> loading pose decoder pretrain '{}'".format(pose_decoder_path))
                    pose_encoder_dict = torch.load(pose_encoder_path)
                    key_list = list(pose_encoder_dict.keys())
                    new_key_list = list(pose_encoder.state_dict().keys())
                    new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
                    pose_encoder_dict = {new_key_map[k]: v for k, v in pose_encoder_dict.items()}
                    pose_encoder.load_state_dict(pose_encoder_dict)

                    pose_decoder_dict = torch.load(pose_decoder_path)
                    key_list = list(pose_decoder_dict.keys())
                    new_key_list = list(pose_decoder.state_dict().keys())
                    new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
                    pose_decoder_dict = {new_key_map[k]: v for k, v in pose_decoder_dict.items()}
                    pose_decoder.load_state_dict(pose_decoder_dict)

        else:
            logger.info(
                '[Attention]: Do not find checkpoint {}'.format(args.depth_pretrain))

    depth_scheduler = optim.lr_scheduler.StepLR(depth_optimizer,scheduler_step_size, 0.1)

    if args.pixor_pretrain:
        if os.path.isfile(args.pixor_pretrain):
            logger.info("=> loading pixor pretrain '{}'".format(
                args.pixor_pretrain))
            checkpoint = torch.load(args.pixor_pretrain)
            pixor.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['lr'] *= 10

        else:
            logger.info(
                '[Attention]: Do not find checkpoint {}'.format(args.pixor_pretrain))

    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.gamma)

    if args.resume:
        logger.info("Resuming...")
        checkpoint_path = osp.join(savepath, args.checkpoint)
        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            pixor.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            depth_decoder.load_state_dict(checkpoint['depth_state_dict'])
            pose_encoder.load_state_dict(checkpoint['pose_encoder_dict'])
            pose_decoder.load_state_dict(checkpoint['pose_decoder_dict'])
            '''
            depth_model.load_state_dict(checkpoint['depth_state_dict'])
            depth_optimizer.load_state_dict(checkpoint['depth_optimizer'])
            depth_scheduler.load_state_dict(checkpoint['depth_scheduler'])
            '''
            start_epoch = checkpoint['epoch'] + 1
            logger.info(
                "Resumed successfully from epoch {}.".format(start_epoch))
        else:
            logger.warning(
                "Model {} not found. "
                "Train from scratch".format(checkpoint_path))
            start_epoch = 0
    else:
        start_epoch = 0

    class_criterion = class_criterion.cuda()
    reg_criterion = reg_criterion.cuda()

    processes = []
    last_eval_epoches = []
    for epoch in range(start_epoch, args.epochs):
        pixor.train()
        ts = time.time()

        #crkim
        encoder.train()
        depth_decoder.train()
        if "M" in args.depth_loss:
            pose_encoder.train()
            pose_decoder.train()
        logger.info("Start epoch {}, depth lr {:.6f} pixor lr {:.7f}".format(
            epoch, depth_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr']))

        avg_class_loss = AverageMeter()
        avg_reg_loss = AverageMeter()
        avg_total_loss = AverageMeter()

        train_metric = utils_func.Metric()

        for iteration, batch in enumerate(train_loader):
            color = batch['color'].cuda()
            color_post = batch['color_post'].cuda()
            color_prev = batch['color_prev'].cuda()
            curr_image = batch['curr_image'].cuda()
            post_image = batch['post_image'].cuda()
            prev_image = batch['prev_image'].cuda()
            intrinsic = batch['intrinsic'].cuda()

            imgL = batch['imgL'].cuda()
            imgR = batch['imgR'].cuda()
            f = batch['f']
            depth_map = batch['depth_map'].cuda()
            idxx = batch['idx']
            h_shift = batch['h_shift']
            ori_shape = batch['ori_shape']
            a_shift = batch['a_shift']
            flip = batch['flip']
            images = batch['image'].cuda()
            img_index = batch['img_index'].cuda()
            bev_index = batch['bev_index'].cuda()

            class_labels = batch['cl'].cuda()
            reg_labels = batch['rl'].cuda()
            depth_loss, depth_map, disp, mask = forward_monodepth_model(imgL, color, depth_map, train_metric, depth_decoder, encoder,mode='TRAIN')
           
            if "D" == args.depth_loss:
                depth_loss = depth_loss
            elif "M" == args.depth_loss:
                norm_image_seq = {'color':color, 'color_post': color_post, 'color_prev':color_prev} #network input
                image_seq = {'curr_image':curr_image, 'post_image':post_image, 'prev_image':prev_image} #warp image
                reprojection_loss = forward_pose_model(args, intrinsic, pose_encoder, pose_decoder, norm_image_seq, image_seq, disp, iteration, mask)
                depth_loss = reprojection_loss *10
            elif "MD" == args.depth_loss:
                norm_image_seq = {'color':color, 'color_post': color_post, 'color_prev':color_prev} #network input
                image_seq = {'curr_image':curr_image, 'post_image':post_image, 'prev_image':prev_image} #warp image
                reprojection_loss = forward_pose_model(args, intrinsic, pose_encoder, pose_decoder, norm_image_seq, image_seq, disp, iteration, mask)
                depth_loss += reprojection_loss *10

            inputs = []
            for i in range(depth_map.shape[0]):
                calib = utils_func.torchCalib(
                    train_data.dataset.get_calibration(idxx[i]), h_shift[i])
                H, W = ori_shape[0][i], ori_shape[1][i]
                depth = depth_map[i][-H:, :W]
                ptc = depth_to_pcl(calib, depth, max_high=1.)
                ptc = calib.lidar_to_rect(ptc[:, 0:3])

                if torch.abs(a_shift[i]).item() > 1e-6:
                    roty = utils_func.roty_pth(a_shift[i]).cuda()
                    ptc = torch.mm(ptc, roty.t())
                voxel = gen_feature_diffused_tensor(
                    ptc, 700, 800, grid_3D_extended, diffused=args.diffused)
                if flip[i] > 0:
                    voxel = torch.flip(voxel, [2])

                inputs.append(voxel)
            inputs = torch.stack(inputs)
            class_outs, reg_outs = pixor(inputs, images,
                                            img_index, bev_index)

            class_outs = class_outs.squeeze(1)
            class_loss, reg_loss, loss = \
                compute_loss(epoch, class_outs, reg_outs,
                    class_labels, reg_labels, class_criterion,
                    reg_criterion, args)
            avg_class_loss.update(class_loss.item())
            avg_reg_loss.update(reg_loss.item() \
                if not isinstance(reg_loss, int) else reg_loss)
            avg_total_loss.update(loss.item())

            optimizer.zero_grad()
            depth_optimizer.zero_grad()
            loss = depth_loss + 0.1*loss
            loss.backward()
            optimizer.step()
            depth_optimizer.step()

            if not isinstance(reg_loss, int):
                reg_loss = reg_loss.item()

            if iteration % args.logevery == 0:
                logger.info("epoch {:d}, iter {:d}, class_loss: {:.5f},"
                    " reg_loss: {:.5f}, loss: {:.5f}".format(epoch,
                        iteration, avg_class_loss.avg, avg_reg_loss.avg,
                        avg_total_loss.avg))
                logger.info("depth loss: {:.5f}".format(depth_loss))
                logger.info(train_metric.print(epoch, iteration))
        
        depth_scheduler.step()
        scheduler.step()

        logger.info("Finish epoch {}, time elapsed {:.3f} s".format(
            epoch, time.time() - ts))

        if (epoch+1) % args.eval_every_epoch == 0 and epoch >= args.start_eval:
            logger.info("Evaluation begins at epoch {}".format(epoch))
            evaluate(eval_data, eval_loader, pixor,encoder, depth_decoder,
                     args.batch_size, gpu=use_gpu, logger=logger,
                     args=args, epoch=epoch, processes=processes,
                     grid_3D_extended=grid_3D_extended)
            if args.run_official_evaluate:
                last_eval_epoches.append((epoch, 7))
                last_eval_epoches.append((epoch, 5))

        if len(last_eval_epoches) > 0:
            for e, iou in last_eval_epoches[:]:
                predicted_results = osp.join(
                    args.saverootpath, args.run_name,
                    'predicted_label_{}'.format(e), 'outputs_{:02d}.txt'.format(iou))
                if osp.exists(predicted_results):
                    with open(predicted_results, 'r') as f:
                        for line in f.readlines():
                            if line.startswith('car_detection_ground AP'):
                                results = [float(num) for num in line.strip(
                                    '\n').split(' ')[-3:]]
                                last_eval_epoches.remove((e, iou))

        if epoch % args.save_every == 0:
            saveto = osp.join(savepath, "checkpoint_{}.pth.tar".format(epoch))
            if "M" in args.depth_loss:
                torch.save({'state_dict': pixor.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'encoder_state_dict': encoder.state_dict(),
                            'depth_state_dict': depth_decoder.state_dict(),
                            'pose_encoder_dict': pose_encoder.state_dict(),
                            'pose_decoder_dict': pose_decoder.state_dict(),
                            'depth_optimizer': depth_optimizer.state_dict(),
                            'depth_scheduler': depth_scheduler.state_dict(),
                            'epoch': epoch
                            }, saveto)
            else:
                torch.save({'state_dict': pixor.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'encoder_state_dict': encoder.state_dict(),
                            'depth_state_dict': depth_decoder.state_dict(),
                            'depth_optimizer': depth_optimizer.state_dict(),
                            'depth_scheduler': depth_scheduler.state_dict(),
                            'epoch': epoch
                            }, saveto)
            logger.info("model saved to {}".format(saveto))
            symlink_force(saveto, osp.join(savepath, "checkpoint.pth.tar"))

    for p in processes:
        if p.wait() != 0:
            logger.warning("There was an error")

    if len(last_eval_epoches) > 0:
        for e, iou in last_eval_epoches[:]:
            predicted_results = osp.join(
                args.saverootpath, args.run_name,
                'predicted_label_{}'.format(e), 'outputs_{:02d}.txt'.format(iou))
            if osp.exists(predicted_results):
                with open(predicted_results, 'r') as f:
                    for line in f.readlines():
                        if line.startswith('car_detection_ground AP'):
                            results = [float(num) for num in line.strip(
                                '\n').split(' ')[-3:]]
                            last_eval_epoches.remove((e, iou))


def evaluate(dataset, data_loader, model, encoder, depth_decoder, batch_size, gpu=False, logger=None,
             args=None, epoch=0, processes=[], grid_3D_extended=None):
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    os.environ['PYTHONHASHSEED'] = str(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # sanity check
    global label_grid
    model.eval()
    encoder.eval()
    depth_decoder.eval()

    if logger == None:
        logger.set_logger("eval")
    logger.info("=> eval lidar_dir: {}".format(dataset.dataset.lidar_dir))
    class_criterion = nn.BCELoss(reduction='none')
    reg_criterion = nn.SmoothL1Loss(reduction='none')
    avg_class_loss = 0
    avg_reg_loss = 0
    avg_loss = 0
    count = 0
    predictions = []

    with torch.no_grad():
        test_metric = utils_func.Metric()
        for batch in tqdm(data_loader) \
            if args.show_eval_progress else data_loader:
            count += 1
            if args.pixor_fusion:
                if not args.e2e:
                    inputs = batch['X'].cuda()
                else:
                    imgL = batch['imgL'].cuda()
                    imgR = batch['imgR'].cuda()
                    #crkim
                    color = batch['color'].cuda()
                    color_post = batch['color_post'].cuda()
                    color_prev = batch['color_prev'].cuda()
                    curr_image = batch['curr_image'].cuda()
                    post_image = batch['post_image'].cuda()
                    prev_image = batch['prev_image'].cuda()
                    intrinsic = batch['intrinsic'].cuda()

                    f = batch['f']
                    depth_map = batch['depth_map'].cuda()
                    idxx = batch['idx']
                    h_shift = batch['h_shift']
                    ori_shape = batch['ori_shape']
                images = batch['image'].cuda()
                img_index = batch['img_index'].cuda()
                bev_index = batch['bev_index'].cuda()
            else:
                inputs = batch['X'].cuda()

            if not args.no_cal_loss:
                class_labels = batch['cl'].cuda()
                reg_labels = batch['rl'].cuda()

            if args.pixor_fusion:
                if not args.e2e:
                    class_outs, reg_outs = pixor(inputs, images,
                                                 img_index, bev_index)
                else:
                    depth_loss, depth_map, _, _ = forward_monodepth_model(imgL, color, depth_map, test_metric, depth_decoder, encoder,  'test')
                    inputs = []
                    for i in range(depth_map.shape[0]):
                        calib = utils_func.torchCalib(
                            dataset.dataset.get_calibration(idxx[i]), h_shift[i])
                        H, W = ori_shape[0][i], ori_shape[1][i]
                        depth = depth_map[i][-H:, :W]

                        #crkim
                        save_depth = np.uint16(depth.clone().cpu().numpy()*256)
                        save_path = os.path.join(args.saverootpath, args.run_name,"pred_depth_"+str(epoch))
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path,"{:06d}.png".format(idxx[i]))
                        cv.imwrite(save_path, save_depth)

                        ptc = depth_to_pcl(calib, depth, max_high=1.)
                        ptc_np = ptc.clone().cpu().numpy()
                        ptc_np = ptc_np.astype(np.float32)

                        #crkim
                        save_path = os.path.join(args.saverootpath, args.run_name, "pred_velodyne_"+str(epoch))
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, "{:06d}.bin".format(idxx[i]))
                        f = open(save_path, 'wb')
                        f.write(ptc_np)
                        f.close()

                        ptc = calib.lidar_to_rect(ptc[:, 0:3])
                        inputs.append(gen_feature_diffused_tensor(
                            ptc, 700, 800, grid_3D_extended, diffused=args.diffused))
                    inputs = torch.stack(inputs)
                    class_outs, reg_outs = model(inputs, images,
                                                 img_index, bev_index)
            else:
                class_outs, reg_outs = model(inputs)

            class_outs = class_outs.squeeze(1)
            if not args.no_cal_loss:
                class_loss, reg_loss, loss = \
                    compute_loss(epoch, class_outs, reg_outs, class_labels,
                                 reg_labels, class_criterion,
                                 reg_criterion, args)

                avg_class_loss += class_loss.item()
                avg_reg_loss += reg_loss.item() if not isinstance(
                    reg_loss, int) else reg_loss
                avg_loss += loss.item()

            if args.gen_predict_file:
                if gpu:
                    class_outs, reg_outs = class_outs.cpu(), reg_outs.cpu()
                predictions += gen_single_prediction_fast(
                    class_outs, reg_outs, label_grid,
                    args.throw_threshold, args.nms_threshold)

    if not args.no_cal_loss:
        avg_class_loss /= count
        avg_reg_loss /= count
        avg_loss /= count
        logger.info("Finish evaluaiton: class_loss: {:.5f}, "
            "reg_loss: {:.5f}, total_loss: {:.5f}".format(
            avg_class_loss, avg_reg_loss, avg_loss))
        logger.info(test_metric.print(epoch, ''))
    else:
        logger.info("Finish evaluaiton!")
    if args.gen_predict_file:
        logger.info("Generating predictions to files")
        logger.info(len(dataset.data))
        savefile_path = osp.join(
            args.saverootpath, args.run_name,
            'predicted_label_{}'.format(epoch))
        predict_kitti_to_file(predictions, dataset.data,
                              osp.join(savefile_path, "data"), dataset.dataset)
        if args.run_official_evaluate:
            label_path = osp.join(args.root_dir, "training/label_2")
            with open(osp.join(savefile_path, "outputs_07.txt"), "w") as f, \
                open(os.devnull, 'w') as FNULL:
                processes.append(subprocess.Popen(
                    [args.evaluate_bin7, label_path, savefile_path],
                    stdout=f, stderr=FNULL))
            with open(osp.join(savefile_path, "outputs_05.txt"), "w") as f, \
                open(os.devnull, 'w') as FNULL:
                processes.append(subprocess.Popen(
                    [args.evaluate_bin5, label_path, savefile_path],
                    stdout=f, stderr=FNULL))

def get_detection_fast(label_grid, class_map, reg_map, throw_threshold=0.5):
    reg_map = reg_map.data.numpy()
    class_map = class_map.data.numpy()

    detections = []
    confidences = []
    valid_idx = class_map >= throw_threshold

    reg_map = reg_map[:, valid_idx]
    class_map = class_map[valid_idx]
    reg_map[2:4] += label_grid[:, valid_idx]

    for i in range(reg_map.shape[1]):
        reg = reg_map[:, i]
        center = (reg[2], reg[3])
        w, h = np.exp(reg[5]), np.exp(reg[4])
        angle = np.arctan2(reg[1], reg[0]) * 180 / np.pi
        detections.append((center, (w, h), angle))
        confidences.append(float(class_map[i]))
    return [detections, confidences]


def gen_single_prediction_fast(class_outs, reg_outs, label_grid,
                               throw_threshold, nms_threshold):
    predictions = []
    for i in range(class_outs.shape[0]):
        prediction = {'box3d_rect': [], 'scores': []}
        [detections, confidences] = get_detection_fast(
            label_grid, class_outs[i], reg_outs[i],
            throw_threshold=throw_threshold)
        indices = np.array(cv.dnn.NMSBoxesRotated(
            detections, confidences, throw_threshold, nms_threshold)).flatten()
        for idx in indices:
            prediction['scores'].append(confidences[idx])
            ret = detections[idx]
            prediction['box3d_rect'].append(
                [ret[2] / 180 * np.pi, ret[0][0],
                ret[0][1], ret[1][1], ret[1][0]])
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    args = parse_args()
    if args.jobid is not None:
        args.run_name = args.run_name + '-' + args.jobid

    if args.shortname is None:
        args.shortname = args.run_name
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = set_logger(name=args.shortname, level=args.loglevel)
    if args.mode == "train":
        logger.info("=> Training mode")
        train(args)
    else:
        logger.info("=> Evaluation mode")

        n_features = 35 if args.no_reflex else 36
        eval_data, eval_loader = get_eval_dataset(args)

        if args.pixor_fusion:
            pixor = PixorNet_Fusion(n_features, groupnorm=args.groupnorm,
                                    resnet_type=args.resnet_type,
                                    image_downscale=args.image_downscale,
                                    resnet_chls=args.resnet_chls)
        else:
            pixor = PixorNet(n_features, groupnorm=args.groupnorm)

        use_gpu = torch.cuda.is_available()
        assert use_gpu, "Please use gpu."
        num_gpu = list(range(torch.cuda.device_count()))

        ts = time.time()
        pixor = pixor.cuda()
        pixor = nn.DataParallel(pixor, device_ids=num_gpu)
        logger.info(
            "Finish cuda loading in {:.3f} s".format(time.time() - ts))

        #crkim
        #----options-----
        num_layers = 18
        #----------------
        parameters_to_train = []
        encoder = networks.ResnetEncoder(num_layers, False)
        encoder = encoder.cuda()
        encoder = nn.DataParallel(encoder).cuda()
        parameters_to_train += list(encoder.parameters())

        depth_decoder = networks.DepthDecoder(encoder.module.num_ch_enc)
        depth_decoder = depth_decoder.cuda()
        depth_decoder = nn.DataParallel(depth_decoder).cuda()
        parameters_to_train += list(depth_decoder.parameters())
        model_dict = encoder.state_dict()
        '''
        pose_encoder = networks.ResnetEncoder(num_layers, weights_init, num_input_images=num_pose_frames).cuda()
        pose_encoder = nn.DataParallel(pose_encoder, device_ids=num_gpu)
        parameters_to_train += list(pose_encoder.parameters())
        
        pose_decoder = networks.PoseDecoder(pose_encoder.module.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).cuda()   
        pose_decoder = nn.DataParallel(pose_decoder, device_ids=num_gpu)            
        parameters_to_train += list(pose_decoder.parameters())
        '''

        '''
        depth_model = PSMNet(maxdepth=80, maxdisp=192, down=args.depth_down)
        depth_model = nn.DataParallel(depth_model, device_ids=num_gpu).cuda()
        '''
        grid_3D_extended = get_3D_global_grid_extended(700, 800, 35).cuda().float()

        if args.eval_ckpt:
            if os.path.isfile(args.eval_ckpt):
                logger.info("=> loading checkpoint '{}'".format(
                    args.eval_ckpt))
                checkpoint = torch.load(args.eval_ckpt)
                #crkim
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                depth_decoder.load_state_dict(checkpoint['depth_state_dict'])
                '''
                depth_model.load_state_dict(checkpoint['depth_state_dict'])
                '''
                pixor.load_state_dict(checkpoint['state_dict'])
                #crkim
                #pose_encoder.load_state_dict(checkpoint['pose_encoder_dict'])
                #pose_decoder.load_state_dict(checkpoint['pose_decoder_dict'])
            else:
                logger.info(
                    '[Attention]: Do not find checkpoint {}'.format(args.eval_ckpt))

        else:
            if args.depth_pretrain:
                #crkim
                encoder_path = os.path.join(args.depth_pretrain, "encoder.pth")
                decoder_path = os.path.join(args.depth_pretrain, "depth.pth")
                if os.path.isfile(encoder_path) and os.path.isfile(decoder_path):
                    logger.info("=> loading depth pretrain '{}'".format(
                        encoder_path))
                    logger.info("=> loading depth pretrain '{}'".format(
                        decoder_path))
                    # crkim
                    #encoder_path = os.path.join(args.depth_pretrain, "encoder.pth")
                    #decoder_path = os.path.join(args.depth_pretrain, "depth.pth")
                    encoder_dict = torch.load(encoder_path)
                    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
                    decoder_dict = torch.load(decoder_path)
                    key_list = list(decoder_dict.keys())
                    new_key_list = list(depth_decoder.state_dict().keys())
                    new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
                    decoder_dict = {new_key_map[k]: v for k, v in decoder_dict.items()}
                    depth_decoder.load_state_dict(decoder_dict)
                    #depth_decoder.load_state_dict(torch.load(decoder_path))
                    '''
                    checkpoint = torch.load(args.depth_pretrain)
                    depth_model.load_state_dict(checkpoint['state_dict'])
                    '''
                    '''
                    if "M" in args.depth_loss:
                        pose_encoder_path = os.path.join(args.depth_pretrain, "pose_encoder.pth")
                        pose_decoder_path = os.path.join(args.depth_pretrain, "pose.pth")
                        if os.path.isfile(pose_encoder_path) and os.path.isfile(pose_decoder_path):
                            logger.info("=> loading pose encoder pretrain '{}'".format(pose_encoder_path))
                            logger.info("=> loading pose decoder pretrain '{}'".format(pose_decoder_path))
                            pose_encoder_dict = torch.load(pose_encoder_path)
                            key_list = list(pose_encoder_dict.keys())
                            new_key_list = list(pose_encoder.state_dict().keys())
                            new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
                            pose_encoder_dict = {new_key_map[k]: v for k, v in pose_encoder_dict.items()}
                            pose_encoder.load_state_dict(pose_encoder_dict)

                            pose_decoder_dict = torch.load(pose_decoder_path)
                            key_list = list(pose_decoder_dict.keys())
                            new_key_list = list(pose_decoder.state_dict().keys())
                            new_key_map = {kl: nkl for kl, nkl in zip(key_list, new_key_list)}
                            pose_decoder_dict = {new_key_map[k]: v for k, v in pose_decoder_dict.items()}
                            pose_decoder.load_state_dict(pose_decoder_dict)
                    '''
            else:
                logger.info(
                    '[Attention]: Do not find checkpoint {}'.format(args.depth_pretrain))

            if args.pixor_pretrain:
                if os.path.isfile(args.pixor_pretrain):
                    logger.info("=> loading pixor pretrain '{}'".format(
                        args.pixor_pretrain))
                    checkpoint = torch.load(args.pixor_pretrain)
                    pixor.load_state_dict(checkpoint['state_dict'])
                else:
                    logger.info(
                        '[Attention]: Do not find checkpoint {}'.format(args.pixor_pretrain))

        processes = []
        logger.info("=> evaluating on {}".format(args.eval_lidar_dir))
        evaluate(eval_data, eval_loader, pixor, encoder, depth_decoder,
            args.batch_size, gpu=use_gpu, logger=logger, args=args,
            epoch=args.eval_save_prefix + "eval",
                 processes=processes, grid_3D_extended=grid_3D_extended)
        for p in processes:
            if p.wait() != 0:
                logger.warning("There was an error")
