from siamese_model import CoattentionNet, CoattentionNet_flow
import numpy as np
import torch
import utils
import os
import os.path as osp
import shutil
from dataloader_scvd import recontruct_test
import cv2


def evaluate_model(trained_model_dir, data_loader, device, training=False, debug=False, segmentation=True, test=False):
    if not training:
        net = CoattentionNet()
        net.load_state_dict(torch.load(trained_model_dir)["model"])
        output_dir = './test_output'
    else:
        net = trained_model_dir
        output_dir = './output_all/'
    net.to(device)
    net.eval()
    mae = 0.0
    mse = 0.0
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not test:
        for idx, (train_last, train_present, last_map, present_map, last_gt_count, present_gt_count, last_fname, present_fname) in enumerate(data_loader):
            last_img, present_img = train_last.to(device), train_present.to(device)
            last_map, present_map = last_map.numpy(), present_map.numpy()

            prediction1, prediction2 = net(last_img, present_img)
            prediction1, prediction2 = prediction1.data.cpu().numpy(), prediction2.data.cpu().numpy()

            prediction1, prediction2 = np.squeeze(prediction1, axis=0), np.squeeze(prediction2, axis=0)
            last_map, present_map = np.squeeze(last_map, axis=0), np.squeeze(present_map, axis=0)
            if not segmentation:
                gt_count1, gt_count2 = np.sum(last_gt_count.numpy()), np.sum(present_gt_count.numpy())
                et_count1, et_count2 = np.sum(prediction1), np.sum(prediction2)
                mae += (abs(gt_count1 - et_count1) + abs(gt_count2 - et_count2))
                mse += ((gt_count1 - et_count1) * (gt_count1 - et_count1) + (gt_count2 - et_count2) * (gt_count2 - et_count2))

            if debug or not training:
                model_name = "siamese"
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                output_dir_now = os.path.join(output_dir, 'density_maps_' + str(model_name))
                if not os.path.exists(output_dir_now):
                    os.mkdir(output_dir_now)
                last_img, present_img = last_img.data.cpu().numpy(), present_img.data.cpu().numpy()
                last_img, present_img = np.squeeze(last_img, axis=0), np.squeeze(present_img, axis=0)
                if segmentation:
                    prediction1, prediction2 = np.where(prediction1 <= 0.5, prediction1, 1), \
                                               np.where(prediction2 <= 0.5, prediction2, 1)
                    prediction1, prediction2 = np.where(prediction1 >= 0.9, prediction1, 0),\
                                               np.where(prediction2 >= 0.9, prediction2, 0)
                utils.save_density_map(last_img, prediction1, output_dir_now, 'output_' + last_fname[0].split('.')[0] + '.png')
                utils.save_density_map(last_img, last_map, output_dir_now, 'gt_' + last_fname[0].split('.')[0] + '.png')
                #utils.save_density_map(present_img, prediction2, output_dir_now, 'output_' + present_fname[0].split('.')[0] + '.png')
                #utils.save_density_map(present_img, present_map, output_dir_now, 'gt_' + present_fname[0].split('.')[0] + '.png')

        if not segmentation:
            mae = mae/(2 * data_loader.__len__())
            mse = np.sqrt(mse / (2 * data_loader.__len__()))

        return mae, mse
    else:
        for idx, (train_last, train_present, last_map, present_map, last_gt_count, present_gt_count, last_fname,
                  present_fname, ori_shape, new_shape) in enumerate(data_loader):
            last_img, present_img = train_last[0].to(device), train_present[0].to(device)
            last_map, present_map = last_map[0].numpy(), present_map[0].numpy()
            if last_img.size(0) > 8:
                prediction1, prediction2 = None, None
                for i in range(0, last_img.size(0)//8+1):
                    prediction_1, prediction_2 = net(last_img[i * 8:min(i * 8 + 8, last_img.size(0))],
                                                     present_img[i * 8:min(i * 8 + 8, last_img.size(0))])
                    prediction_1, prediction_2 = prediction_1.data.cpu().numpy(), prediction_2.data.cpu().numpy()
                    if prediction1 is None:
                        prediction1, prediction2 = prediction_1, prediction_2
                    else:
                        prediction1, prediction2 = np.concatenate((prediction1, prediction_1), axis=0), \
                                                   np.concatenate((prediction2, prediction_2), axis=0)
            else:
                prediction1, prediction2 = net(last_img, present_img)
                prediction1, prediction2 = prediction1.data.cpu().numpy(), prediction2.data.cpu().numpy()

            ori_shape, new_shape = tuple(ori_shape[0].data.cpu().numpy()), tuple(new_shape[0].data.cpu().numpy())
            new_prediction1, new_prediction2 = np.zeros((prediction1.shape[0], 1, 473, 473)), \
                                               np.zeros((prediction2.shape[0], 1, 473, 473))
            for i in range(prediction1.shape[0]):
                new_prediction1[i, 0], new_prediction2[i, 0] = cv2.resize(prediction1[i, 0], (473, 473)), \
                                                               cv2.resize(prediction2[i, 0], (473, 473))
                new_prediction1[i, 0] = new_prediction1[i, 0] * (
                        (prediction1.shape[-2] * prediction1.shape[-1]) / (473 * 473))
                new_prediction2[i, 0] = new_prediction2[i, 0] * (
                            (prediction2.shape[-2] * prediction2.shape[-1]) / (473 * 473))

            last_img, present_img = last_img.data.cpu().numpy(), present_img.data.cpu().numpy()

            ori_last_img, prediction1 = recontruct_test(last_img, new_prediction1, ori_shape, new_shape)
            ori_present_img, prediction2 = recontruct_test(present_img, new_prediction2, ori_shape, new_shape)
            _, last_map = recontruct_test(last_img, last_map, ori_shape, new_shape)
            _, present_map = recontruct_test(present_img, present_map, ori_shape, new_shape)

            if not segmentation:
                #gt_count1, gt_count2 = np.sum(last_gt_count.numpy()), np.sum(present_gt_count.numpy())
                gt_count1, gt_count2 = np.sum(last_map), np.sum(present_map)
                et_count1, et_count2 = np.sum(prediction1), np.sum(prediction2)
                mae += abs(gt_count1 - et_count1)
                mse += (gt_count1 - et_count1) * (gt_count1 - et_count1)

            if debug or not training:
                model_name = "siamese"
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                output_dir_now = os.path.join(output_dir, 'density_maps_' + str(model_name))
                if not os.path.exists(output_dir_now):
                    os.mkdir(output_dir_now)
                if segmentation:
                    prediction1, prediction2 = np.where(prediction1 <= 0.5, prediction1, 1), \
                                               np.where(prediction2 <= 0.5, prediction2, 1)
                    prediction1, prediction2 = np.where(prediction1 >= 0.9, prediction1, 0), \
                                               np.where(prediction2 >= 0.9, prediction2, 0)
                utils.save_density_map(ori_last_img, prediction1, output_dir_now,
                                       'output_' + last_fname[0].split('.')[0] + '.png')
                utils.save_density_map(ori_last_img, last_map, output_dir_now, 'gt_' + last_fname[0].split('.')[0] + '.png')

        if not segmentation:
            mae = mae / data_loader.__len__()
            mse = np.sqrt(mse / data_loader.__len__())

        return mae, mse


def generate_motion_flow(trained_model_dir, data_loader, device, training=False):
    if not training:
        net = CoattentionNet_flow()
        model_dict = net.state_dict()
        pretrained_dict = torch.load(trained_model_dir)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    else:
        net = trained_model_dir
    net.to(device)
    net.eval()
    if os.path.exists('./flow_test'):
        shutil.rmtree('./flow_test')
    os.mkdir('flow_test')

    for idx, (train_last_transforms, train_present_transforms, _, _, last_path, present_path) in enumerate(data_loader):
        last_img, present_img = train_last_transforms.float().to(device), train_present_transforms.float().to(device)

        prediction1, prediction2 = net(last_img, present_img)
        prediction1, prediction2 = prediction1.data.cpu().numpy(), prediction2.data.cpu().numpy()

        flow = np.abs(prediction2 - prediction1)
        flow = np.squeeze(flow, axis=0)
        last_path = '/'.join(last_path[0].split('/')[-1].split('\\')[-2:])
        present_path = '/'.join(present_path[0].split('/')[-1].split('\\')[-2:])
        last_dir, present_dir = osp.join("./flow_test", last_path.split('/')[0][:-4]), \
                                osp.join("./flow_test", present_path.split('/')[0][:-4])
        last_path, present_path = last_path.split('/')[-1][:-4], present_path.split('/')[-1][:-4]
        if not os.path.exists(last_dir):
            os.mkdir(last_dir)
        if not os.path.exists(present_dir):
            os.mkdir(present_dir)
        utils.save_density_map(flow, last_dir, fname="{}_{}.png".format(last_path, present_path.split('_')[-1]))
