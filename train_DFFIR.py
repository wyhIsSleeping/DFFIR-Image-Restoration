# -*- coding: utf-8 -*-
import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
import numpy as np
import time
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from utils.dataset_utils import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset
from net.model_OKM_refined import DFFIR
import subprocess
from torch.utils.data import DataLoader
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
import clip

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--save_item', type=int, default=2000,
                    help='save model per every N item')
parser.add_argument('--init_epoch', type=int, default=2,
                    help='if finetune model, set the initial epoch')

parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                     help='save parameter dir')

parser.add_argument('--gpu', type=str, default="0",
                    help='GPUs') 

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--pretrained_1', type=str, default='', 
        help='training loss')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=1,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')


parser.add_argument('--data_file_dir', type=str, default='data_dir/',  
                    help='where clean images of denoising saves.') 
parser.add_argument('--denoise_dir', type=str, default='./data/Train/Denoise/',
                    help='where clean images of denoising saves.')           
parser.add_argument('--derain_dir', type=str, default='./data/Train/Derain/',
                    help='where training images of deraining saves.')          
parser.add_argument('--dehaze_dir', type=str, default='./data/Train/Dehaze/',
                    help='where training images of dehazing saves.')          

parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")


parser.add_argument('--denoise_path', type=str, default="test/denoise/", 
                    help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="test/derain/", 
                    help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", 
                    help='save path of test hazy images')

parser.add_argument('--output_path', type=str, default="./output", 
                    help='output save path')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
checkpoint_dir = os.path.join(args.save_dir, 'model_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

final_results_dir = os.path.join(args.save_dir, "final_results")
os.makedirs(final_results_dir, exist_ok=True)

log_dir = os.path.join(args.save_dir, "logs_train")
os.makedirs(log_dir, exist_ok=True)

psnr_max = 10

device = torch.device(f"cuda:{args.cuda}")

clip_model, _ = clip.load("ViT-B/32", device=device)
for param in clip_model.parameters():
    param.requires_grad = False  

inputext = ["Gaussian noise with a standard deviation of 15","Gaussian noise with a standard deviation of 25"
            ,"Gaussian noise with a standard deviation of 50","Rain degradation with rain lines"
            ,"Hazy degradation with normal haze"] 

denoise_splits = ["bsd68/"]
derain_splits = ["Rain100L/"]
denoise_tests = []
derain_tests = []
base_path = args.denoise_path

args.derain_path = args.derain_path + "Rain100L/"

for i in denoise_splits:
    args.denoise_path = os.path.join(base_path, i)
    denoise_testset = DenoiseTestDataset(args)
    denoise_tests.append(denoise_testset)

def train(train_loader, model, optimizer, epoch, epoch_total, criterionL1):
    loss_sum = 0
    losses = AverageMeter()
    
    writer = SummaryWriter(log_dir)
    psnr_tqdm = 10
    ssim_tqdm = 0.009
    loss_tqdm = 0.0

    model.train()
    start_time = time.time()
    global psnr_max

    loop_train = tqdm((train_loader), total=len(train_loader), leave=False, colour="magenta") 
    for i, ([clean_name, de_id], degrad_patch, clean_patch) in enumerate(loop_train):

        input_var = Variable(degrad_patch.to(device))
        target_var = Variable(clean_patch.to(device))

        result = [clean_name, de_id]
        img_id = result[1]
        img_id = img_id.tolist()  
        text_prompt_list = [inputext[idx] for idx in img_id]

        text_token = clip.tokenize(text_prompt_list).to(device) 
        text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  

        output = model(input_var, text_code)
        loss = criterionL1(output, target_var)
        loss_sum += loss.item()
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 10 == 0) and (i != 0):
            loss_avg = loss_sum / 10
            loss_sum = 0.0 
            loss_tqdm = loss_avg                                                   
            writer.add_scalar("train_loss", loss.item(), i)
            start_time = time.time()     
        if (i % args.save_item == 0) and (i != 0): 
            
            psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50, psnr_rain, ssim_rain, psnr_haze, ssim_haze = test(model, criterionL1, save_images=False)
            
            psnr_avr = (psnr_n15 + psnr_n25 + psnr_n50 + psnr_rain + psnr_haze) / 5
            ssim_avr = (ssim_n15 + ssim_n25 + ssim_n50 + ssim_rain + ssim_haze) / 5
            psnr_tqdm = psnr_avr
            ssim_tqdm  = ssim_avr

            if psnr_avr > psnr_max - 0.0001:
                psnr_max = max(psnr_avr, psnr_max)
              
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                 'checkpoint_epoch_V1_{:0>4}_{}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'
                                 .format(epoch, i//args.save_item, psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                                         psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr)))
                print(f'Model saved at epoch {epoch}, item {i} with PSNR: {psnr_avr:.4f}')
            else:
              
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_item_{i}.pth.tar')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    checkpoint_path)
                
        loop_train.set_description(f'training->epoch:[{epoch}/{args.epochs}],item:[{i}/{len(train_loader)}]') 
        loop_train.set_postfix(loss=loss_tqdm, psnr=f'{psnr_tqdm:.4f}', ssim=f'{ssim_tqdm:.4f}')        
    writer.close()
    return losses.avg

def test(model, criterion, save_images=False):
    model.eval()
    
    if not save_images:
       
        psnr_g15, ssim_g15 = test_Denoise(model, denoise_tests[0], sigma=15, text_prompt=inputext[0], save_images=False)
        psnr_g25, ssim_g25 = test_Denoise(model, denoise_tests[0], sigma=25, text_prompt=inputext[1], save_images=False)
        psnr_g50, ssim_g50 = test_Denoise(model, denoise_tests[0], sigma=50, text_prompt=inputext[2], save_images=False)
        
        derain_set = DerainDehazeDataset(args, addnoise=False, sigma=15)
        psnr_rain, ssim_rain = test_Derain_Dehaze(model, derain_set, task="derain", text_prompt=inputext[3], save_images=False)
        psnr_haze, ssim_haze = test_Derain_Dehaze(model, derain_set, task="dehaze", text_prompt=inputext[4], save_images=False)
    else:
       
        print("Saving final results to:", final_results_dir)
       
        denoise_final_dir = os.path.join(final_results_dir, "denoise")
        derain_final_dir = os.path.join(final_results_dir, "derain")
        dehaze_final_dir = os.path.join(final_results_dir, "dehaze")
        
        os.makedirs(denoise_final_dir, exist_ok=True)
        os.makedirs(derain_final_dir, exist_ok=True)
        os.makedirs(dehaze_final_dir, exist_ok=True)
        
        psnr_g15, ssim_g15 = test_Denoise(model, denoise_tests[0], sigma=15, text_prompt=inputext[0], 
                                          save_images=True, output_dir=os.path.join(denoise_final_dir, "sigma_15"))
        psnr_g25, ssim_g25 = test_Denoise(model, denoise_tests[0], sigma=25, text_prompt=inputext[1], 
                                          save_images=True, output_dir=os.path.join(denoise_final_dir, "sigma_25"))
        psnr_g50, ssim_g50 = test_Denoise(model, denoise_tests[0], sigma=50, text_prompt=inputext[2], 
                                          save_images=True, output_dir=os.path.join(denoise_final_dir, "sigma_50"))
        
        derain_set = DerainDehazeDataset(args, addnoise=False, sigma=15)
        psnr_rain, ssim_rain = test_Derain_Dehaze(model, derain_set, task="derain", text_prompt=inputext[3], 
                                                  save_images=True, output_dir=derain_final_dir)
        psnr_haze, ssim_haze = test_Derain_Dehaze(model, derain_set, task="dehaze", text_prompt=inputext[4], 
                                                  save_images=True, output_dir=dehaze_final_dir)
    
    return psnr_g15, ssim_g15, psnr_g25, ssim_g25, psnr_g50, ssim_g50, psnr_rain, ssim_rain, psnr_haze, ssim_haze

def test_Denoise(net, dataset, sigma=15, text_prompt="", save_images=False, output_dir=None):
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0) 
    psnr = AverageMeter()
    ssim = AverageMeter() 
    text_token = clip.tokenize(text_prompt).to(device) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
    
    with torch.no_grad():
        desc = f"Denoise sigma={sigma}" + (" (saving images)" if save_images else "")
        pbar = tqdm(testloader, desc=desc, leave=False, colour="green")
        for ([clean_name], degrad_patch, clean_patch) in pbar:
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
        
            restored = net(degrad_patch, text_code)          
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)    
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            if save_images and output_dir:
                filename = clean_name[0].replace('.png', '')
                save_path = os.path.join(output_dir, f"{filename}_sigma{sigma}_psnr{temp_psnr:.2f}.png")
                save_image_tensor(restored, save_path)
            
            pbar.set_postfix(PSNR=f'{psnr.avg:.2f}', SSIM=f'{ssim.avg:.4f}')
        
    return psnr.avg, ssim.avg

def test_Derain_Dehaze(net, dataset, task="derain", text_prompt="", save_images=False, output_dir=None):
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(device) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32) 
    
    with torch.no_grad():
        desc = f"{task.capitalize()} Testing" + (" (saving images)" if save_images else "")
        pbar = tqdm(testloader, desc=desc, leave=False, colour="blue")
        for ([degraded_name], degrad_patch, clean_patch) in pbar:
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            restored = net(degrad_patch, text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            if save_images and output_dir:
                filename = degraded_name[0].replace('.png', '')
                save_path = os.path.join(output_dir, f"{filename}_psnr{temp_psnr:.2f}.png")
                save_image_tensor(restored, save_path)
            
            pbar.set_postfix(PSNR=f'{psnr.avg:.2f}', SSIM=f'{ssim.avg:.4f}')
    
    return psnr.avg, ssim.avg

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(0)
    torch.manual_seed(0)
    
    torch.cuda.set_device(device)
    
    model = DFFIR(device=device)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*50)
    print("MODEL PARAMETER COUNT")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print("="*50)
    
    criterionL1 = nn.L1Loss().to(device)

    if os.path.exists(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch))):
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)),
                                map_location=device)
        print('==> loading existing model:',
              os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler.load_state_dict(model_info['scheduler'])
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        cur_epoch = args.init_epoch

    if args.pretrained_1:
        if os.path.isfile(args.pretrained_1):
            print("=> loading model '{}'".format(args.pretrained_1))
            model_pretrained = torch.load(args.pretrained_1,
                                          map_location=device)
                                          
            pretrained_dict = model_pretrained['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained_1)) 

    train_dataset = PromptTrainDataset(args)              
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    print('Dataset loaded successfully')
    print('Training data path:', args.denoise_dir)
    print('Model save path:', args.save_dir)
    print('Final results will be saved to:', final_results_dir)
  
    print('Running initial test (no images saved)...')
    psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50, psnr_rain, ssim_rain, psnr_haze, ssim_haze = test(model, criterionL1, save_images=False) 
    psnr_avr = (psnr_n15 + psnr_n25 + psnr_n50 + psnr_rain + psnr_haze) / 5
    ssim_avr = (ssim_n15 + ssim_n25 + ssim_n50 + ssim_rain + ssim_haze) / 5
    print('Initial test completed - Average PSNR: {:.4f}, Average SSIM: {:.4f}'.format(psnr_avr, ssim_avr))           

    for epoch in range(cur_epoch, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}:')
        loss = train(train_loader, model, optimizer, epoch, args.epochs + 1, criterionL1)
        scheduler.step()

        if epoch % args.save_epoch == 0:
            print('Running validation test...')
         
            psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50, psnr_rain, ssim_rain, psnr_haze, ssim_haze = test(model, criterionL1, save_images=False)
            psnr_avr = (psnr_n15 + psnr_n25 + psnr_n50 + psnr_rain + psnr_haze) / 5
            ssim_avr = (ssim_n15 + ssim_n25 + ssim_n50 + ssim_rain + ssim_haze) / 5
            
            print(f'Epoch {epoch} - Loss: {loss:.5f}, Average PSNR: {psnr_avr:.4f}, Average SSIM: {ssim_avr:.4f}')
            
            if psnr_avr > psnr_max - 0.01:
                psnr_max = max(psnr_avr, psnr_max)
                checkpoint_path = os.path.join(args.save_dir,
                                'checkpoint_epoch_V1_{:0>4}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'
                                .format(epoch, psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                                        psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    checkpoint_path)
                print(f'Model saved (new best PSNR: {psnr_max:.4f}) at {checkpoint_path}')
            else:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth.tar')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    checkpoint_path)
                print(f'Model saved at {checkpoint_path}')

    print("\n" + "="*50)
    print("Training completed! Running final test and saving results...")
    print("="*50)
    
    final_psnr_n15, final_ssim_n15, final_psnr_n25, final_ssim_n25, final_psnr_n50, final_ssim_n50, \
    final_psnr_rain, final_ssim_rain, final_psnr_haze, final_ssim_haze = test(model, criterionL1, save_images=True)
    
    final_psnr_avr = (final_psnr_n15 + final_psnr_n25 + final_psnr_n50 + final_psnr_rain + final_psnr_haze) / 5
    final_ssim_avr = (final_ssim_n15 + final_ssim_n25 + final_ssim_n50 + final_ssim_rain + final_ssim_haze) / 5
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Average PSNR: {final_psnr_avr:.4f}")
    print(f"Average SSIM: {final_ssim_avr:.4f}")
    print(f"Results saved to: {final_results_dir}")
    print("="*50)
    
    summary_path = os.path.join(final_results_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FINAL TEST RESULTS SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Denoise (sigma=15): PSNR={final_psnr_n15:.4f}, SSIM={final_ssim_n15:.4f}\n")
        f.write(f"Denoise (sigma=25): PSNR={final_psnr_n25:.4f}, SSIM={final_ssim_n25:.4f}\n")
        f.write(f"Denoise (sigma=50): PSNR={final_psnr_n50:.4f}, SSIM={final_ssim_n50:.4f}\n")
        f.write(f"Derain (Rain100L): PSNR={final_psnr_rain:.4f}, SSIM={final_ssim_rain:.4f}\n")
        f.write(f"Dehaze (SOTS): PSNR={final_psnr_haze:.4f}, SSIM={final_ssim_haze:.4f}\n")
        f.write("-"*50 + "\n")
        f.write(f"Average PSNR: {final_psnr_avr:.4f}\n")
        f.write(f"Average SSIM: {final_ssim_avr:.4f}\n")
    
    print(f"Results summary saved to: {summary_path}")