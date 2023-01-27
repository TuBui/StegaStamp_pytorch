import argparse
import os
import glob
import PIL
from PIL import Image, ImageOps
import bchlib
from time import time
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tools.image_dataset import ImageDataset
from tools.eval_metrics import unormalise, compute_psnr, compute_ssim, compute_mse, compute_lpips
from tools.augment_imagenetc import RandomImagenetC
# from models import StegaStampEncoder, StegaStampDecoder
import lpips

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

def load_models(args):

    encoder = torch.load(args.encoder_path)
    encoder.eval()
    FINGERPRINT_SIZE = 100

    decoder = torch.load(args.decoder_path)
    decoder.eval()

    return encoder, decoder, FINGERPRINT_SIZE


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # model
    device = torch.device("cpu") if int(args.cuda) == -1 else torch.device("cuda:0")
    HideNet, RevealNet, FINGERPRINT_SIZE = load_models(args)
    HideNet, RevealNet = HideNet.to(device), RevealNet.to(device)

    # data
    tform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.ToTensor(),
            ]
        )
    dataset = ImageDataset(args.data_dir, args.data_list, transform=tform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(dataset)

    # test
    lpips_alex = lpips.LPIPS(net='alex').to(device)
    noise = RandomImagenetC(1, 5, 'train')
    noise_ids = noise.corrupt_ids 
    score_lpips, score_ssim, score_psnr, score_mse = [], [], [], []
    bit_acc = {i: [] for i in noise_ids}
    bit_acc[-1] = []

    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x['x'].to(device)
            secret = generate_random_fingerprints(FINGERPRINT_SIZE, batch_size=x.shape[0]).to(device)
            stegos_ = HideNet((secret, x)) + x  # [0, 1]
            stegos = stegos_ * 2. - 1  # [-1, 1]

            x_unorm = unormalise(x)  # [0, 255], uint8, NHWC
            stegos_unorm = unormalise(stegos)

            # eval stego quality: SSIM, PSNR, MSE, LPIPS
            score_lpips.append(compute_lpips(x, stegos, lpips_alex))
            score_ssim.append(compute_ssim(x_unorm, stegos_unorm))
            score_psnr.append(compute_psnr(x_unorm, stegos_unorm))
            score_mse.append(compute_mse(x_unorm, stegos_unorm))

            # perturb stego
            noise_id = np.random.choice(noise_ids)
            stegos_perturbed = [tform(noise(Image.fromarray(im), noise_id)) for im in stegos_unorm]
            stegos_perturbed = torch.stack(stegos_perturbed).to(device)

            secret = secret.cpu().numpy()

            # predict secret
            secret_pred = RevealNet(stegos_).cpu().numpy().round()
            bit_acc[-1].append(np.mean(secret == secret_pred, axis=1))

            # predict secret perturbed
            secret_pred = RevealNet(stegos_perturbed).cpu().numpy().round()
            bit_acc[noise_id].append(np.mean(secret == secret_pred, axis=1))

    score_lpips, score_ssim, score_psnr, score_mse = [np.concatenate(x) for x in [score_lpips, score_ssim, score_psnr, score_mse]]
    bit_acc = {i: np.concatenate(x) for i, x in bit_acc.items()}

    print(f"lpips: {score_lpips.mean():.4f} +- {score_lpips.std():.4f}")
    print(f"ssim: {score_ssim.mean():.4f} +- {score_ssim.std():.4f}")
    print(f"psnr: {score_psnr.mean():.4f} +- {score_psnr.std():.4f}")
    print(f"mse: {score_mse.mean():.4f} +- {score_mse.std():.4f}")

    out = dict(score_lpips=score_lpips, score_ssim=score_ssim, score_psnr=score_psnr, score_mse=score_mse)
    for i in bit_acc:
        name = 'clean' if i==-1 else noise.method_names[i]
        print(f"bit_acc {name}: {bit_acc[i].mean():.4f} +- {bit_acc[i].std():.4f}")
        out[f'bit_acc_{name}'] = bit_acc[i]

    bit_acc_noise = np.concatenate([val for i, val in bit_acc.items() if i!=-1])
    print(f"bit_acc noise: {bit_acc_noise.mean():.4f} +- {bit_acc_noise.std():.4f}")

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    np.savez(os.path.join(args.output, 'results.npz'), **out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-enc',
        "--encoder_path", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/stegastamp_pt/s100/checkpoints/encoder_80000.pth', help="Path to trained StegaStamp encoder."
    )
    parser.add_argument('-d', "--data_dir", type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/mia/datasets/ffhq256_lmdb2', help="Directory with images.")
    parser.add_argument('-l', '--data_list', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/mia/datasets/ffhq256_lmdb2/val.csv', help="Path to list of images.")

    parser.add_argument('-o', 
        "--output", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/stegastamp_pt/s100/results', help="output directory."
    )
    parser.add_argument(
        "--image_resolution", type=int, default=400, help="Height and width of square images."
    )
    parser.add_argument('-dec',
        "--decoder_path",
        default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/stegastamp_pt/s100/checkpoints/decoder_80000.pth',
        help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    main(args)
