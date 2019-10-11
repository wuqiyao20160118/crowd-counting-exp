import numpy as np
import cv2
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from config import config
import utils
import time

random.seed(int(time.time()))


class TrainDataLoader(Dataset):
    def __init__(self, last_transforms, present_transforms, root_dir="./data/scvd/train", training=True):
        self.max_inter = config.max_inter
        self.name = root_dir.split('/')[-2]
        self.ret = {}
        self.root_dir = root_dir
        self.sub_class_dir = os.listdir(root_dir)
        self.last_transforms = last_transforms
        self.present_transforms = present_transforms
        self.count = 0
        self.total = 0
        self.training = training
        all_folders = os.listdir(root_dir)
        for single_folder in all_folders:
            fnames = os.listdir(os.path.join(root_dir, single_folder))
            self.total += len(fnames) // 2
            assert isinstance(self.total, int)

    def _pick_img_pairs(self, index_of_subclass):

        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'

        video_name = self.sub_class_dir[index_of_subclass]
        video_dir = os.path.join(self.root_dir, video_name)
        frame_info = utils.get_frame_annotation(video_dir)
        video_num = len(frame_info)
        if self.max_inter >= video_num - 1:
            self.max_inter = video_num // 2

        last_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num - 1)
        present_index = np.clip(random.choice(range(1, max(2, self.max_inter))) + last_index, 0, video_num - 1)

        last_img_path, present_img_path = frame_info[last_index]['img'], frame_info[present_index]['img']
        last_gt, present_gt = frame_info[last_index]['annotation'], frame_info[present_index]['annotation']
        ori_last_gt, ori_present_gt = frame_info[last_index]['ori_annotation'], \
                                      frame_info[present_index]['ori_annotation']
        self.kernel = utils._gaussian_kernel(sigma=4.0, kernel_size=15)

        # load infomation of template and detection
        self.ret['last_img_path'] = last_img_path
        self.ret['present_img_path'] = present_img_path
        self.last_gt = last_gt
        self.present_gt = present_gt
        self.ori_last_gt = ori_last_gt
        self.ori_present_gt = ori_present_gt

    def _tranform(self):
        self.ret['train_last_transforms'] = self.last_transforms(self.ret['last_img'])
        self.ret['train_present_transforms'] = self.present_transforms(self.ret['present_img'])

    def open(self):

        '''template'''
        #last_img = cv2.imread(self.ret['last_img_path']) #if you use cv2.imread you can not open .JPEG format
        last_img = Image.open(self.ret['last_img_path'])
        last_img = np.array(last_img)
        present_img = Image.open(self.ret['present_img_path'])
        present_img = np.array(present_img)
        if np.random.rand(1) < config.gray_ratio:
            last_img = cv2.cvtColor(last_img, cv2.COLOR_RGB2GRAY)
            last_img = cv2.cvtColor(last_img, cv2.COLOR_GRAY2RGB)
            present_img = cv2.cvtColor(present_img, cv2.COLOR_RGB2GRAY)
            present_img = cv2.cvtColor(present_img, cv2.COLOR_GRAY2RGB)

        if self.training and self.name != "scvd":
            temp_last, temp_present = np.array([]), np.array([])
            while temp_last.shape[0] == 0 or temp_present.shape[0] == 0:
                last_img, present_img, temp_last, temp_present, last_paste_box = \
                    utils.crop_image(last_img, present_img, self.ori_last_gt, self.ori_present_gt)
            print("Crop done!")
            temp_last[:, 0], temp_last[:, 1] = (temp_last[:, 0] + temp_last[:, 1]) / 2, \
                                               (temp_last[:, 2] + temp_last[:, 3]) / 2
            temp_present[:, 0], temp_present[:, 1] = (temp_present[:, 0] + temp_present[:, 1]) / 2, \
                                                     (temp_present[:, 2] + temp_present[:, 3]) / 2
            self.ret['last_gt'] = temp_last[:, :2]
            self.ret['present_gt'] = temp_present[:, :2]
        else:
            ori_h, ori_w = last_img.shape[0], last_img.shape[1]
            last_img, present_img = cv2.resize(last_img, (471, 471)), cv2.resize(present_img, (471, 471))
            scale_h, scale_w = last_img.shape[0] / ori_h, last_img.shape[1] / ori_w
            self.last_gt[:, 0], self.last_gt[:, 1] = self.last_gt[:, 0] * scale_w, self.last_gt[:, 1] * scale_h
            self.present_gt[:, 0], self.present_gt[:, 1] = self.present_gt[:, 0] * scale_w, \
                                                           self.present_gt[:, 1] * scale_h
            self.ret['last_gt'], self.ret['present_gt'] = self.last_gt, self.present_gt

        self.ret['last_img'] = last_img  # H, W, C
        self.ret['present_img'] = present_img
        last_map = utils._create_heatmap(last_img.shape, last_img.shape[0:2], self.ret['last_gt'], self.kernel)
        present_map = utils._create_heatmap(present_img.shape, present_img.shape[0:2], self.ret['present_gt'], self.kernel)
        self.ret['last_map'] = np.expand_dims(last_map, axis=0)
        self.ret['present_map'] = np.expand_dims(present_map, axis=0)

    def __getitem__(self, index):
        index = random.choice(range(len(self.sub_class_dir)))

        self._pick_img_pairs(index)
        self.open()
        self._tranform()
        self.count += 1

        return self.ret['train_last_transforms'], self.ret['train_present_transforms'], \
               self.ret['last_map'], self.ret['present_map'], self.ret['last_img_path'], self.ret['present_img_path']

    def __len__(self):
        return self.total


class ValDataLoader(Dataset):
    def __init__(self, last_transforms, present_transforms, root_dir="./data/scvd/test", training=False):
        self.max_inter = config.max_inter
        self.name = root_dir.split('/')[-2]
        self.ret = {}
        self.root_dir = root_dir
        self.sub_class_dir = os.listdir(root_dir)
        self.last_transforms = last_transforms
        self.present_transforms = present_transforms
        self.count = 0
        self.total = 0
        self.video_num = []
        self.training = training
        all_folders = os.listdir(root_dir)
        for single_folder in all_folders:
            fnames = os.listdir(os.path.join(root_dir, single_folder))
            self.video_num.append(self.total)
            self.total += len(fnames) // 2 - 1
            assert isinstance(self.total, int)

    def _pick_img_pairs(self, index_of_subclass, idx):

        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'

        video_name = self.sub_class_dir[index_of_subclass]
        video_dir = os.path.join(self.root_dir, video_name)
        frame_info = utils.get_frame_annotation(video_dir)
        video_num = len(frame_info)
        if self.max_inter >= video_num - 1:
            self.max_inter = video_num // 2

        last_index = np.clip(idx, 0, video_num - 1)
        present_index = np.clip(idx+1, 0, video_num - 1)

        last_img_path, present_img_path = frame_info[last_index]['img'], frame_info[present_index]['img']
        last_gt, present_gt = frame_info[last_index]['annotation'], frame_info[present_index]['annotation']
        ori_last_gt, ori_present_gt = frame_info[last_index]['ori_annotation'], \
                                      frame_info[present_index]['ori_annotation']
        self.kernel = utils._gaussian_kernel(sigma=4.0, kernel_size=15)

        # load infomation of template and detection
        self.ret['last_img_path'] = last_img_path
        self.ret['present_img_path'] = present_img_path
        self.last_gt = last_gt
        self.present_gt = present_gt
        self.ori_last_gt = ori_last_gt
        self.ori_present_gt = ori_present_gt

    def _tranform(self):
        self.ret['train_last_transforms'] = self.last_transforms(self.ret['last_img'])
        self.ret['train_present_transforms'] = self.present_transforms(self.ret['present_img'])

    def open(self):

        '''template'''
        #last_img = cv2.imread(self.ret['last_img_path']) #if you use cv2.imread you can not open .JPEG format
        last_img = Image.open(self.ret['last_img_path'])
        last_img = np.array(last_img)
        present_img = Image.open(self.ret['present_img_path'])
        present_img = np.array(present_img)
        if np.random.rand(1) < config.gray_ratio:
            last_img = cv2.cvtColor(last_img, cv2.COLOR_RGB2GRAY)
            last_img = cv2.cvtColor(last_img, cv2.COLOR_GRAY2RGB)
            present_img = cv2.cvtColor(present_img, cv2.COLOR_RGB2GRAY)
            present_img = cv2.cvtColor(present_img, cv2.COLOR_GRAY2RGB)

        if self.training and self.name != "scvd":
            temp_last, temp_present = np.array([]), np.array([])
            while temp_last.shape[0] == 0 or temp_present.shape[0] == 0:
                last_img, present_img, temp_last, temp_present, last_paste_box = \
                    utils.crop_image(last_img, present_img, self.ori_last_gt, self.ori_present_gt)
            print("Crop done!")
            temp_last[:, 0], temp_last[:, 1] = (temp_last[:, 0] + temp_last[:, 1]) / 2, \
                                               (temp_last[:, 2] + temp_last[:, 3]) / 2
            temp_present[:, 0], temp_present[:, 1] = (temp_present[:, 0] + temp_present[:, 1]) / 2, \
                                                     (temp_present[:, 2] + temp_present[:, 3]) / 2
            self.ret['last_gt'] = temp_last[:, :2]
            self.ret['present_gt'] = temp_present[:, :2]
        else:
            ori_h, ori_w = last_img.shape[0], last_img.shape[1]
            last_img, present_img = cv2.resize(last_img, (471, 471)), cv2.resize(present_img, (471, 471))
            scale_h, scale_w = last_img.shape[0] / ori_h, last_img.shape[1] / ori_w
            self.last_gt[:, 0], self.last_gt[:, 1] = self.last_gt[:, 0] * scale_w, self.last_gt[:, 1] * scale_h
            self.present_gt[:, 0], self.present_gt[:, 1] = self.present_gt[:, 0] * scale_w, \
                                                           self.present_gt[:, 1] * scale_h
            self.ret['last_gt'], self.ret['present_gt'] = self.last_gt, self.present_gt

        self.ret['last_img'] = last_img  # H, W, C
        self.ret['present_img'] = present_img
        last_map = utils._create_heatmap(last_img.shape, last_img.shape[0:2], self.ret['last_gt'], self.kernel)
        present_map = utils._create_heatmap(present_img.shape, present_img.shape[0:2], self.ret['present_gt'], self.kernel)
        self.ret['last_map'] = np.expand_dims(last_map, axis=0)
        self.ret['present_map'] = np.expand_dims(present_map, axis=0)

    def __getitem__(self, index):
        idx = 0
        for i in range(len(self.video_num)):
            if self.video_num[i] > index:
                break
            idx = i
        total_num = self.video_num[idx]
        resident = index - total_num
        self._pick_img_pairs(idx, resident)
        self.open()
        self._tranform()
        self.count += 1

        return self.ret['train_last_transforms'], self.ret['train_present_transforms'], \
               self.ret['last_map'], self.ret['present_map'], self.ret['last_img_path'], self.ret['present_img_path']

    def __len__(self):
        return self.total


if __name__ == "__main__":
    from torchvision import transforms
    from custom_transforms import Normalize, ToTensor
    from torch.utils import data
    import os.path as osp
    train_data = TrainDataLoader(transforms.Compose([ToTensor()]), transforms.Compose([ToTensor()]), root_dir="./data/scvd/test")
    data_loader = data.DataLoader(train_data, batch_size=1)
    for idx, (_, _, last, present, l_path, p_path) in enumerate(data_loader):
        last_path = "/".join(l_path[0].split('/')[-1].split('\\')[-2:])
        print(osp.join("./test", last_path.split('/')[0]))
        print(last_path.split('/')[-1][:-4])
        if idx > 1:
            break

