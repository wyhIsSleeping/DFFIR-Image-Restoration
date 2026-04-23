# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from utils.dataset_utils import DerainDehazeDataset
from net.model_OKM_refined import ChannelShuffle_skip_textguaid
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.image_utils import crop_img

def parse_args():
    parser = argparse.ArgumentParser(description='Dehaze Test (Consistent with Training Logic)')
    parser.add_argument('--gpu', type=str, default="0", help='GPU device ID')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--pretrained_1', type=str, default='./checkpoints/checkpoint.pth.tar', 
                        help='path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--output_path', type=str, default='./results/dehaze/', 
                        help='output save path')
    parser.add_argument('--dehaze_hazy_dir', type=str, default='./data/test/hazy/', 
                        help='directory of hazy images')
    parser.add_argument('--dehaze_gt_dir', type=str, default='./data/test/GT/', 
                        help='directory of ground truth images')
    parser.add_argument('--index', type=int, default=None, 
                        help='Only process specific image index (0-based)')
    return parser.parse_args()

def init_model(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.set_device(args.cuda)
    
    model = ChannelShuffle_skip_textguaid(device=args.cuda)
    model = model.cuda(args.cuda)
    model.eval()
    
    if os.path.isfile(args.pretrained_1):
        print(f"=> Loading trained model from {args.pretrained_1}")
        checkpoint = torch.load(args.pretrained_1, map_location=torch.device(f'cuda:{args.cuda}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> Model loaded successfully")
    else:
        raise FileNotFoundError(f"=> No model found at {args.pretrained_1}")
    
    clip_model, _ = clip.load("ViT-B/32", device=args.cuda)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    dehaze_prompt = "Hazy degradation with normal haze"
    
    return model, clip_model, dehaze_prompt

def dehaze_test(args, model, clip_model, dehaze_prompt):
    test_dataset = DerainDehazeDataset(
        args, 
        task="dehaze", 
        addnoise=False, 
        sigma=None
    )
    
    # 如果指定了 index，只取对应的图片
    if args.index is not None:
        if args.index < len(test_dataset):
            indices = [args.index]
            print(f"Processing only image index {args.index}")
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                pin_memory=True, 
                shuffle=False, 
                num_workers=0,
                sampler=torch.utils.data.SubsetRandomSampler(indices)
            )
        else:
            raise Exception(f"Index {args.index} out of range. Total images: {len(test_dataset)}")
    else:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            pin_memory=True, 
            shuffle=False, 
            num_workers=0
        )
    
    psnr = AverageMeter()
    ssim = AverageMeter()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    text_token = clip.tokenize(dehaze_prompt).to(args.cuda)
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Dehaze Testing", colour="blue")
        for ([degraded_name], degrad_patch, clean_patch) in pbar:
            degrad_patch, clean_patch = degrad_patch.cuda(args.cuda), clean_patch.cuda(args.cuda)
            
            restored = model(degrad_patch, text_code)
            
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            filename = os.path.splitext(degraded_name[0])[0]
            save_path = os.path.join(args.output_path, f"{filename}_PSNR_{temp_psnr:.2f}dB_SSIM_{temp_ssim:.4f}.png")
            save_image_tensor(restored, save_path)
            
            pbar.set_postfix(PSNR=f'{psnr.avg:.2f}', SSIM=f'{ssim.avg:.4f}')
            print(f"\nImage: {filename}")
            print(f"PSNR: {temp_psnr:.2f} dB, SSIM: {temp_ssim:.4f}")
    
    if psnr.count > 0:
        summary_path = os.path.join(args.output_path, "results_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("DEHAZE TEST RESULTS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Average PSNR: {psnr.avg:.4f} dB\n")
            f.write(f"Average SSIM: {ssim.avg:.4f}\n")
            f.write(f"Number of images: {psnr.count}\n")
        print(f"\nResults saved to {summary_path}")
    
    return psnr.avg, ssim.avg

if __name__ == '__main__':
    args = parse_args()
    model, clip_model, dehaze_prompt = init_model(args)
    avg_psnr, avg_ssim = dehaze_test(args, model, clip_model, dehaze_prompt)
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")