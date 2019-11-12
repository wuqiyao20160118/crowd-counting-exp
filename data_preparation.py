from utils import _gaussian_kernel, _create_heatmap, get_frame_annotation
import numpy as np
import pandas as pd
import os
import cv2
import random
import time

random.seed(int(time.time()))


def create_training_data(data_path, training=True):
    if training:
        base_data = "train"
    else:
        base_data = "val"
    if not os.path.exists('./data/scvd_processed'):
        os.mkdir('./data/scvd_processed')
        os.mkdir('./data/scvd_processed/' + base_data)
        os.mkdir('./data/scvd_processed/' + base_data + '_den')
    if not os.path.exists('./data/scvd_processed/' + base_data):
        os.mkdir('./data/scvd_processed/' + base_data)
        os.mkdir('./data/scvd_processed/' + base_data + '_den')
    all_folders = os.listdir(data_path)
    for single_folder in all_folders:
        if not os.path.exists(os.path.join('./data/scvd_processed/' + base_data, single_folder)):
            os.mkdir(os.path.join('./data/scvd_processed/' + base_data, single_folder))
        if not os.path.exists(os.path.join('./data/scvd_processed/' + base_data + '_den', single_folder)):
            os.mkdir(os.path.join('./data/scvd_processed/' + base_data + '_den', single_folder))
        im_save_path = os.path.join('./data/scvd_processed/' + base_data, single_folder)
        den_save_path = os.path.join('./data/scvd_processed/' + base_data + '_den', single_folder)
        fnames = os.listdir(os.path.join(data_path, single_folder))
        for j in range(0, 4):
            flag = False
            for fname in fnames:
                xml_name = os.path.join(os.path.splitext(fname)[0] + '.png')
                annos = get_frame_annotation(os.path.join(data_path, single_folder, xml_name))["annotation"]
                img = cv2.imread(os.path.join(data_path, single_folder, xml_name))
                img = img.astype(np.float32, copy=False)
                ori_shape = img.shape[:2]
                ori_h, ori_w = ori_shape[0], ori_shape[1]
                wn2, hn2 = 473 / 2, 473 / 2
                if ori_w <= 2 * wn2:
                    img = cv2.resize(img, (int(2*wn2)+1, ori_h))
                    annos[:, 0] = annos[:, 0] * 2 * wn2 / ori_w
                if ori_h <= 2 * hn2:
                    img = cv2.resize(img, (ori_w, int(2*hn2)+1))
                    annos[:, 1] = annos[:, 1] * 2 * hn2 / ori_h
                ori_shape = img.shape[:2]
                ori_h, ori_w = ori_shape[0], ori_shape[1]
                a_w, b_w = wn2, ori_w - wn2
                a_h, b_h = hn2, ori_h - hn2
                im_density = _create_heatmap(ori_shape, ori_shape, annotation_points=annos,
                                             kernel=_gaussian_kernel(4.0, 15))

                if not flag:
                    x = int((b_w - a_w) * random.random() + a_w)
                    y = int((b_h - a_h) * random.random() + a_h)
                    flag = True
                x1, y1 = x - int(wn2), y - int(hn2)
                x2, y2 = x + int(wn2) + 1, y + int(hn2) + 1
                im_sampled = img[y1:y2, x1:x2, :]
                im_density_sampled = im_density[y1:y2, x1: x2]
                annos = annos[annos[:, 0] > x1, :]
                annos = annos[annos[:, 0] < x2, :]
                annos = annos[annos[:, 1] > y1, :]
                annPoints_sampled = annos[annos[:, 1] < y2, :]
                annPoints_sampled[:, 0] = annPoints_sampled[:, 0] - x1
                annPoints_sampled[:, 1] = annPoints_sampled[:, 1] - y1
                cv2.imwrite(os.path.join(im_save_path, xml_name[:-4]+"_"+str(j)+".png"), im_sampled)
                bbox = pd.DataFrame(im_density_sampled)
                bbox.to_csv(os.path.join(den_save_path, xml_name[:-4]+"_"+str(j)+".csv"), header=0, index=0)


if __name__ == "__main__":
    create_training_data("./data/scvd/train")
    create_training_data("./data/scvd/test", training=False)
