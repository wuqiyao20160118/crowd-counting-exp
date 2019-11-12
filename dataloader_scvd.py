import numpy as np
import cv2
import os
import torch
from torch.autograd import Variable
import pandas as pd
import random
from torch.utils.data import Dataset
from config import config
import time
import operator
import utils

random.seed(int(time.time()))
DOWNSAMPLE = 4


def np_to_variable(x, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        with torch.no_grad():
            v = Variable(torch.from_numpy(x).type(dtype), requires_grad=False)
    return v


class TrainDataLoader(Dataset):
    def __init__(self, device, gt_path="./data/scvd_processed/train_den", data_path="./data/scvd_processed/train", training=True, segmentation=False):
        self.max_inter = config.max_inter
        self.name = data_path.split('/')[-2]
        self.ret = {}
        self.data_files = []
        self.folder_num = []
        self.data_path = data_path
        self.gt_path = gt_path
        self.training = training
        self.device = device
        self.segmentation = segmentation
        all_folders = os.listdir(data_path)
        total = 0
        for single_folder in all_folders:
            fnames = os.listdir(os.path.join(data_path, single_folder))
            fnames.sort()
            self.folder_num.append(total)
            total += len(fnames)
            for fname in fnames:
                if not operator.eq(fname, 'roi.csv'):
                    self.data_files.append(os.path.join(os.path.splitext(fname)[0] + '.png'))
        self.folder_num.append(total)
        self.shuffle = True if training else False
        if self.shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)

    def open(self, fname, flip=False):
        copy_of_fname = fname
        split_results_of_fname = copy_of_fname.split('.')
        possible_folder_name = split_results_of_fname[0]
        folder_name = possible_folder_name[:-6] + '_screenshot'

        img = cv2.imread(os.path.join(self.data_path, folder_name, fname))
        img = img.astype(np.float32, copy=False)
        if flip:
            img = cv2.flip(img, 1)

        img1, img2, img3 = cv2.split(img)
        img = img.reshape((3, img.shape[0], img.shape[1]))
        img[0] = img1
        img[1] = img2
        img[2] = img3

        den = pd.read_csv(os.path.join(os.path.join(self.gt_path, folder_name), os.path.splitext(fname)[0] + '.csv'),
                          sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        if flip:
            den = cv2.flip(den, 1)
        gt_count = np.sum(den)
        den = cv2.resize(den, (den.shape[1] // DOWNSAMPLE, den.shape[0] // DOWNSAMPLE))
        den = den * ((den.shape[1] * DOWNSAMPLE * den.shape[0] * DOWNSAMPLE) / (den.shape[0] * den.shape[1]))

        if self.segmentation:
            den = np.where(den <= 0.001, den, 1)

        den = den.reshape((1, den.shape[0], den.shape[1]))
        return img, den, gt_count

    def __getitem__(self, index):
        while (index+1) in self.folder_num or (index+2) in self.folder_num or (index+3) in self.folder_num or (index+4) in self.folder_num:
            index = int(random.choice(range(0, self.folder_num[-1] - 4)))
        if random.random() < 0.6:
            flip = False
        else:
            flip = True
        total_num = np.where(np.array(self.folder_num) > index)[0][0]
        video_num = self.folder_num[total_num]
        last_fname = self.data_files[index]
        last_img, last_den, last_gt_count = self.open(last_fname, flip=flip)
        last_img = np_to_variable(last_img, is_training=True)
        last_den = np_to_variable(last_den, is_training=True)

        present_index = int(random.choice(range(1, max(2, (video_num - index) // 4)))) * 4 + index
        present_fname = self.data_files[present_index]
        present_img, present_den, present_gt_count = self.open(present_fname, flip=flip)
        present_img = np_to_variable(present_img, is_training=True)
        present_den = np_to_variable(present_den, is_training=True)

        return last_img, present_img, last_den, present_den, last_gt_count, present_gt_count, last_fname, present_fname

    def __len__(self):
        return self.num_samples


class ValDataLoader(Dataset):
    def __init__(self, device, gt_path="./data/scvd_processed/val_den", data_path="./data/scvd_processed/val", training=False, segmentation=False):
        self.max_inter = config.max_inter
        self.name = data_path.split('/')[-2]
        self.ret = {}
        self.data_files = []
        self.folder_num = []
        self.data_path = data_path
        self.gt_path = gt_path
        self.training = training
        self.device = device
        self.segmentation = segmentation
        all_folders = os.listdir(data_path)
        total = 0
        for single_folder in all_folders:
            fnames = os.listdir(os.path.join(data_path, single_folder))
            fnames.sort()
            self.folder_num.append(total)
            total += len(fnames)
            for fname in fnames:
                if not operator.eq(fname, 'roi.csv'):
                    self.data_files.append(os.path.join(os.path.splitext(fname)[0] + '.png'))
        self.folder_num.append(total)
        self.shuffle = False
        if self.shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)

    def open(self, fname):
        copy_of_fname = fname
        split_results_of_fname = copy_of_fname.split('.')
        possible_folder_name = split_results_of_fname[0]
        folder_name = possible_folder_name[:-6] + '_screenshot'

        img = cv2.imread(os.path.join(self.data_path, folder_name, fname))
        img = img.astype(np.float32, copy=False)

        img1, img2, img3 = cv2.split(img)
        img = img.reshape((3, img.shape[0], img.shape[1]))
        img[0] = img1
        img[1] = img2
        img[2] = img3

        den = pd.read_csv(os.path.join(os.path.join(self.gt_path, folder_name), os.path.splitext(fname)[0] + '.csv'),
                          sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        gt_count = np.sum(den)
        den = cv2.resize(den, (den.shape[1] // DOWNSAMPLE, den.shape[0] // DOWNSAMPLE))
        den = den * ((den.shape[1] * DOWNSAMPLE * den.shape[0] * DOWNSAMPLE) / (den.shape[0] * den.shape[1]))

        if self.segmentation:
            den = np.where(den <= 0.001, den, 1)

        den = den.reshape((1, den.shape[0], den.shape[1]))
        return img, den, gt_count

    def __getitem__(self, index):
        if (index+1) in self.folder_num or (index+2) in self.folder_num or (index+3) in self.folder_num or (index+4) in self.folder_num:
            index -= 4
        last_fname = self.data_files[index]
        last_img, last_den, last_gt_count = self.open(last_fname)
        last_img = np_to_variable(last_img, is_training=False)
        last_den = np_to_variable(last_den, is_training=False)

        present_index = index + 4
        present_fname = self.data_files[present_index]
        present_img, present_den, present_gt_count = self.open(present_fname)
        present_img = np_to_variable(present_img, is_training=False)
        present_den = np_to_variable(present_den, is_training=False)

        return last_img, present_img, last_den, present_den, last_gt_count, present_gt_count, last_fname, present_fname

    def __len__(self):
        return self.num_samples


class TestDataLoader(Dataset):
    def __init__(self, device, data_path="./data/scvd/test", training=False):
        self.max_inter = config.max_inter
        self.name = data_path.split('/')[-2]
        self.ret = {}
        self.data_files = []
        self.folder_num = []
        self.data_path = data_path
        self.training = training
        self.device = device
        all_folders = os.listdir(data_path)
        total = 0
        for single_folder in all_folders:
            fnames_all = os.listdir(os.path.join(data_path, single_folder))
            fnames = []
            for f in fnames_all:
                if "png" in f:
                    fnames.append(f)
            fnames.sort()
            self.folder_num.append(total)
            total += len(fnames)
            for fname in fnames:
                if not operator.eq(fname, 'roi.csv'):
                    self.data_files.append(os.path.join(os.path.splitext(fname)[0] + '.png'))
        self.folder_num.append(total)
        self.shuffle = False
        if self.shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)

    def open(self, fname):
        copy_of_fname = fname
        split_results_of_fname = copy_of_fname.split('.')
        possible_folder_name = split_results_of_fname[0]
        folder_name = possible_folder_name[:-4] + '_screenshot'

        img = cv2.imread(os.path.join(self.data_path, folder_name, fname))
        img = img.astype(np.float32, copy=False)
        ori_shape = img.shape[:2]
        h, w = ori_shape[0], ori_shape[1]
        if h < 473 and w < 473:
            img = cv2.resize(img, (473, 473))
            h, w = 473, 473
        elif h < 473:
            img = cv2.resize(img, (w, 473))
            h = 473
        elif w < 473:
            img = cv2.resize(img, (473, h))
            w = 473

        #img = cv2.resize(img, (473, 473))
        img1, img2, img3 = cv2.split(img)
        img = img.reshape((3, img.shape[0], img.shape[1]))
        img[0] = img1
        img[1] = img2
        img[2] = img3

        annos = utils.get_frame_annotation(os.path.join(self.data_path, folder_name, fname))['annotation']
        kernel = utils._gaussian_kernel(4.0, 15)
        den = utils._create_heatmap(ori_shape, (h, w), annos, kernel=kernel)
        gt_count = np.sum(den)
        den = den.reshape((1, den.shape[0], den.shape[1]))

        ori_shape = (h, w)

        # compute padding
        padding_h = 473 - max(h % 473, (h - 473 // 2) % 473)
        padding_w = 473 - max(w % 473, (w - 473 // 2) % 473)

        # add padding
        img = np.concatenate((img, np.zeros((img.shape[0], padding_h, img.shape[2]))), axis=1)
        den = np.concatenate((den, np.zeros((den.shape[0], padding_h, den.shape[2]))), axis=1)
        img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], padding_w))), axis=2)
        den = np.concatenate((den, np.zeros((den.shape[0], den.shape[1], padding_w))), axis=2)

        # create batches
        imgs, dens = None, None
        _, h, w = img.shape
        new_shape = (h, w)
        disp_height, disp_width = 473 // 2, 473 // 2
        for i in range(0, h - 473 + 1, disp_height):
            for j in range(0, w - 473 + 1, disp_width):
                chunk_img = img[:, i:i + 473, j:j + 473]
                chunk_den = den[:, i:i + 473, j:j + 473]
                if imgs is None:
                    imgs = np.expand_dims(chunk_img, axis=0)
                else:
                    imgs = np.concatenate((imgs, np.expand_dims(chunk_img, axis=0)), axis=0)
                if dens is None:
                    dens = np.expand_dims(chunk_den, axis=0)
                else:
                    dens = np.concatenate((dens, np.expand_dims(chunk_den, axis=0)), axis=0)
            chunk_img = img[:, i:i + 473, -473:]
            chunk_den = den[:, i:i + 473, -473:]
            if imgs is None:
                imgs = np.expand_dims(chunk_img, axis=0)
            else:
                imgs = np.concatenate((imgs, np.expand_dims(chunk_img, axis=0)), axis=0)
            if dens is None:
                dens = np.expand_dims(chunk_den, axis=0)
            else:
                dens = np.concatenate((dens, np.expand_dims(chunk_den, axis=0)), axis=0)
        chunk_img = img[:, -473:, -473:]
        chunk_den = den[:, -473:, -473:]
        if imgs is None:
            imgs = np.expand_dims(chunk_img, axis=0)
        else:
            imgs = np.concatenate((imgs, np.expand_dims(chunk_img, axis=0)), axis=0)
        if dens is None:
            dens = np.expand_dims(chunk_den, axis=0)
        else:
            dens = np.concatenate((dens, np.expand_dims(chunk_den, axis=0)), axis=0)

        img, den = imgs, dens
        return img, den, gt_count, ori_shape, new_shape

    def __getitem__(self, index):
        last_fname = self.data_files[index]
        last_img, last_den, last_gt_count, last_ori_shape, last_new_shape = self.open(last_fname)
        last_img = np_to_variable(last_img, is_training=False)
        last_den = np_to_variable(last_den, is_training=False)

        if (index + 1) in self.folder_num:
            present_index = index
        else:
            present_index = index + 1
        present_fname = self.data_files[present_index]
        present_img, present_den, present_gt_count, present_ori_shape, present_new_shape = self.open(present_fname)
        present_img = np_to_variable(present_img, is_training=False)
        present_den = np_to_variable(present_den, is_training=False)
        assert last_ori_shape == present_ori_shape

        return last_img, present_img, last_den, present_den, last_gt_count, present_gt_count, last_fname, present_fname,\
               np.array(list(last_ori_shape)), np.array(list(last_new_shape))

    def __len__(self):
        return self.num_samples


def recontruct_test(img_batch, den_batch, orig_shape, new_shape):
    disp_height, disp_width = 473 // 2, 473 // 2
    img = np.zeros((3, new_shape[0], new_shape[1]))
    den = np.zeros(new_shape)
    cnt = np.zeros(new_shape)
    ind = 0

    for i in range(0, new_shape[0] - 473 + 1, disp_height):
        for j in range(0, new_shape[1] - 473 + 1, disp_width):
            img[:, i:i + 473, j:j + 473] = img_batch[ind, :]
            den[i:i + 473, j:j + 473] += den_batch[ind, 0]
            cnt[i:i + 473, j:j + 473] += 1
            ind += 1
        img[:, i:i + 473, -473:] = img_batch[ind, :]
        den[i:i + 473, -473:] += den_batch[ind, 0]
        cnt[i:i + 473, -473:] += 1
        ind += 1
    img[:, -473:, -473:] = img_batch[ind, :]
    den[-473:, -473:] += den_batch[ind, 0]
    cnt[-473:, -473:] += 1
    ind += 1
    den /= cnt
    # crop to original shape
    img = img[:, :orig_shape[0], :orig_shape[1]].reshape((3, orig_shape[0], orig_shape[1]))
    den = den[:orig_shape[0], :orig_shape[1]].reshape((1, orig_shape[0], orig_shape[1]))
    return img, den


if __name__ == "__main__":
    from torch.utils import data
    device = torch.device('cpu')
    test_data = TestDataLoader(device=device)
    data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
    for idx, (last_img, present_img, last_den, present_den, _, _, last_name, present_name, ori_shape, new_shape) in enumerate(data_loader):
        print(last_name)
        print(present_name)
        print(last_den[0].size())
        print(ori_shape[0].data.cpu().numpy())

        last_img, last_den = recontruct_test(last_img[0], last_den[0], ori_shape[0].data.cpu().numpy(), new_shape[0].data.cpu().numpy())
        print(last_img.shape)  # C, H, W
        last_img = np.transpose(last_img, (1, 2, 0))
        cv2.imwrite("test.png", img=last_img)
        print("-----------------------")
        if idx > 1:
            break

