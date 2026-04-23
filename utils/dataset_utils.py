import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
import torchvision.transforms as transforms

from glob import glob


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        # ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy_file = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside.txt")

        with open(hazy_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    # 移除可能的多余空格和换行符
                    path = path.strip()

                    # 检查路径是否存在
                    if os.path.exists(path):
                        temp_ids.append(path)
                    else:
                        # 尝试使用相对路径
                        relative_path = os.path.join(self.args.dehaze_dir, os.path.basename(path))
                        if os.path.exists(relative_path):
                            temp_ids.append(relative_path)
                        else:
                            print(f"警告: 有雾图像路径不存在，跳过: {path}")

        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.hazy_ids = self.hazy_ids   # ← 添加这行
        random.shuffle(self.hazy_ids)  # ← 可选：打乱顺序
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        try:
            hazy_basename = os.path.basename(hazy_name)

            # 读取两个列表文件
            hazy_list_path = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside.txt")
            nonhazy_list_path = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside2.txt")

            # 确保文件存在
            if not os.path.exists(hazy_list_path) or not os.path.exists(nonhazy_list_path):
                return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

            with open(hazy_list_path, 'r') as f:
                hazy_list = [line.strip() for line in f if line.strip()]

            with open(nonhazy_list_path, 'r') as f:
                nonhazy_list = [line.strip() for line in f if line.strip()]

            # 按索引返回对应的无雾图像路径
            if hazy_basename in hazy_list:
                index = hazy_list.index(hazy_basename)
                if index < len(nonhazy_list):
                    gt_path = nonhazy_list[index]
                    # 确保路径是绝对路径或正确构建相对路径
                    if not os.path.isabs(gt_path):
                        # 假设GT图像在相对于dehaze_dir的某个位置
                        gt_path = os.path.join(os.path.dirname(self.args.dehaze_dir), gt_path)
                    return gt_path

            # 如果找不到，使用路径替换
            return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

        except Exception as e:
            print(f"获取无雾图像路径出错: {e}")
            return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids

        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class PromptTrainDataset_SP(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset_SP, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        # ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.hazy_ids = self.hazy_ids * 200

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids

        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        # print("de_id",de_id)

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        if de_id < 3:
            de_id = 0  # 3中噪声等级退化类型合并 均返回‘Noise’
        elif de_id == 3:
            de_id = 1
        elif de_id == 4:
            de_id = 2
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


# 5D 'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5 lowlight
class PromptTrainDataset5D(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset5D, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5,
                        'lowlight': 6}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:  # ---------
            self._init_blur_ids()
        if 'lowlight' in self.de_type:  # ---------
            self._init_lol_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "noisy", "denoise.txt")
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [os.path.join(self.args.denoise_dir, id_) for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_rs_ids(self):
        rs_file = os.path.join(self.args.data_file_dir, "rainy", "rainTrain.txt")
        temp_ids = []
        with open(rs_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    full_path = os.path.join(self.args.derain_dir, path)
                    if os.path.exists(full_path):
                        temp_ids.append(full_path)
                    else:
                        print(f"Warning: Rainy image path does not exist, skipping: {full_path}")
        
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 80

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy_file = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside.txt")

        with open(hazy_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    # 移除可能的多余空格和换行符
                    path = path.strip()

                    # 检查路径是否存在
                    if os.path.exists(path):
                        temp_ids.append(path)
                    else:
                        # 尝试使用相对路径
                        relative_path = os.path.join(self.args.dehaze_dir, os.path.basename(path))
                        if os.path.exists(relative_path):
                            temp_ids.append(relative_path)
                        else:
                            print(f"Warning: Hazy image path does not exist, skipping: {path}")

        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_blur_ids(self):  # ------
        temp_ids = []
        blur_file = os.path.join(self.args.data_file_dir, "gopro", "train_gopro.txt")

        with open(blur_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    # 安全拼接路径
                    blur_path = os.path.join(self.args.deblur_dir, 'blur', filename)
                    if os.path.exists(blur_path):
                        temp_ids.append(blur_path)
                    else:
                        print(f"Warning: Blur image path does not exist, skipping: {blur_path}")

        self.blur_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]
        self.blur_ids = self.blur_ids * 30

        self.blur_counter = 0

        self.num_blur = len(self.blur_ids)
        print("Total blur Ids : {}".format(self.num_blur))

    def _init_lol_ids(self):  # ------
        temp_ids = []
        lol_file = os.path.join(self.args.data_file_dir, "lol", "train_lol.txt")

        with open(lol_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    # 安全拼接路径
                    lowlight_path = os.path.join(self.args.lowlight_dir, 'low', filename)
                    if os.path.exists(lowlight_path):
                        temp_ids.append(lowlight_path)
                    else:
                        print(f"Warning: Lowlight image path does not exist, skipping: {lowlight_path}")

        self.lol_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids]
        self.lol_ids = self.lol_ids * 60

        self.lol_counter = 0

        self.num_lol = len(self.lol_ids)
        print("Total lol Ids : {}".format(self.num_lol))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        try:
            hazy_basename = os.path.basename(hazy_name)

            # 读取两个列表文件
            hazy_list_path = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside.txt")
            nonhazy_list_path = os.path.join(self.args.data_file_dir, "hazy", "hazy_outside2.txt")

            # 确保文件存在
            if not os.path.exists(hazy_list_path) or not os.path.exists(nonhazy_list_path):
                return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

            with open(hazy_list_path, 'r') as f:
                hazy_list = [line.strip() for line in f if line.strip()]

            with open(nonhazy_list_path, 'r') as f:
                nonhazy_list = [line.strip() for line in f if line.strip()]

            # 按索引返回对应的无雾图像路径
            if hazy_basename in hazy_list:
                index = hazy_list.index(hazy_basename)
                if index < len(nonhazy_list):
                    gt_path = nonhazy_list[index]
                    # 确保路径是绝对路径或正确构建相对路径
                    if not os.path.isabs(gt_path):
                        # 假设GT图像在相对于dehaze_dir的某个位置
                        gt_path = os.path.join(os.path.dirname(self.args.dehaze_dir), gt_path)
                    return gt_path

            # 如果找不到，使用路径替换
            return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

        except Exception as e:
            print(f"获取无雾图像路径出错: {e}")
            return hazy_name.replace("hazy", "GT").replace("Hazy", "GT")

    def _get_sharp_name(self, blur_name):  # ------get no blur
        sharp_name = blur_name.split("blur")[0] + 'sharp/' + blur_name.split('/')[-1]
        return sharp_name

    def _get_light_name(self, lol_name):  # ------ get no blur
        light_name = lol_name.split("low")[0] + 'high/' + lol_name.split('/')[-1]
        return light_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids

        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids

        if "deblur" in self.de_type:  # -------
            self.sample_ids += self.blur_ids

        if "lowlight" in self.de_type:  # ------
            self.sample_ids += self.lol_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # blur Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_sharp_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 6:
                # lowlight enhancement
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_light_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)

class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_, tile=128, tile_overlap=0):
        sigma_dict = {0: 0, 1: 15, 2: 25, 3: 50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:  # derain
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:  # dehaze
            self.ids = []
            hazy_dir = "/home/ubuntu/data/OTS/test/hazy/"
            name_list = os.listdir(hazy_dir)
            self.ids += [os.path.join(hazy_dir, id_) for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:  # derain
            gt_name = degraded_name.replace("input/", "target/no")
        elif self.task_idx == 1:  # dehaze
          
            hazy_basename = os.path.basename(degraded_name)
            gt_dir = "/home/ubuntu/data/OTS/test/GT/"
            
            gt_path = os.path.join(gt_dir, hazy_basename)
            
            if not os.path.exists(gt_path):
               
                base_name = hazy_basename.replace('_hazy', '').replace('hazy_', '')
                gt_path = os.path.join(gt_dir, base_name)
          
            if not os.path.exists(gt_path):
                gt_path = degraded_name.replace("hazy", "GT")
                
            gt_name = gt_path
            
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        # 核心修改：将base=16改为base=8，确保尺寸是8的整数倍
        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        # 同样修改clean_img的裁剪基数
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length



#  gopro
class DeblurTestDataset(Dataset):
    def __init__(self, args, addnoise=False, sigma=None):
        super(DeblurTestDataset, self).__init__()
        self.ids = []
        self.args = args

        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self._init_input_ids()

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.deblur_path + 'blur/')
        self.ids += [self.args.deblur_path + 'blur/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        gt_name = degraded_name.replace("blur", "sharp")
        return gt_name

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-1]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


#  LOL
class LOLTestDataset(Dataset):
    def __init__(self, args, addnoise=False, sigma=None):
        super(LOLTestDataset, self).__init__()
        self.ids = []
        self.args = args

        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self._init_input_ids()

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.lowlight_path + 'low/')
        self.ids += [self.args.lowlight_path + 'low/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        gt_name = degraded_name.replace("low", "high")
        return gt_name

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=8)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=8)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-1]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length



 