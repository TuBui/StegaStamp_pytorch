import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import argparse
import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips


def inference(encoder, image, secret, out_path):
    """
    image: [B,C,H,W], B=1
    secret: 
    """
    with torch.no_grad():
        residual = encoder((secret, image))
        encoded = image + residual
        residual = residual.cpu()
        encoded = encoded.cpu()
        encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))
        im = Image.fromarray(encoded)
        im.save(out_path)


def main(args):
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    dataset = StegaData(args.data_dir, args.data_list, args.secret_size, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = model.StegaStampEncoder()
    decoder = model.StegaStampDecoder(secret_size=args.secret_size)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = 400
    width = 400
    # prep inference
    cover_sample, secret_sample = dataset[0]
    Image.fromarray(np.array(cover_sample*255, dtype=np.uint8).transpose((1,2,0))).save(os.path.join(log_path, 'sample_org.png'))
    cover_sample = cover_sample.unsqueeze(0).cuda()
    secret_sample = secret_sample.unsqueeze(0).cuda()


    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator, lpips_alex,
                                                                            secret_input, image_input,
                                                                            args.l2_edge_gain, args.borders,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            yuv_scales, args, global_step, writer)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            if global_step % 100 == 0:
                print(f'Iter #{global_step}: Loss = {loss:.4f}, secret loss = {secret_loss:.4f}, D_loss = {D_loss:.4f}, bit_acc = {bit_acc:.4f}, str_acc = {str_acc:.4f}', flush=True)
            if global_step % 1000 ==0:
                out_path = os.path.join(log_path, f'sample_{global_step}.jpg')
                inference(encoder, cover_sample, secret_sample, out_path)
            if global_step % 10000 == 0:
                torch.save(encoder, os.path.join(args.checkpoints_path, f"encoder_{global_step}.pth"))
                torch.save(decoder, os.path.join(args.checkpoints_path , f"decoder_{global_step}.pth"))
                torch.save(discriminator, os.path.join(args.checkpoints_path , f"discriminator_{global_step}.pth"))
    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--data_dir', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/mia/datasets/ffhq256_lmdb2')
    parser.add_argument('--data_list', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/mia/datasets/ffhq256_lmdb2/train.csv')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--no_jpeg', action='store_true')
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('-o', '--output', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/stegastamp_pt/test')
    opt = parser.parse_args()

    with open('cfg/setting.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    # args.data_dir = opt.data_dir
    # args.data_list = opt.data_list
    # args.exp_name = opt.exp_name
    # args.secret_size = opt.secret_size
    args.update(**vars(opt))
    args.checkpoints_path = os.path.join(opt.output, 'checkpoints/')
    args.logs_path = os.path.join(opt.output, "logs/")
    args.saved_models = os.path.join(opt.output, 'saved_models')
    print(args)

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    if not os.path.exists(args.saved_models):
        os.makedirs(args.saved_models)
    main(args)
