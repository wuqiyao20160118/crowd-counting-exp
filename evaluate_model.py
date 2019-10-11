from siamese_model import CoattentionNet, CoattentionNet_flow
import numpy as np
import torch
import utils
import os
import os.path as osp
import shutil


def evaluate_model(trained_model_dir, data_loader, device, training=False, debug=False):
    if not training:
        net = CoattentionNet(training=training)
        net.load_state_dict(torch.load(trained_model_dir))
    else:
        net = trained_model_dir
    net.to(device)
    net.eval()
    mae = 0.0
    mse = 0.0
    temp = 0
    if os.path.exists('./test'):
        shutil.rmtree('./test')
    os.mkdir('test')

    for idx, (train_last_transforms, train_present_transforms, last_map, present_map, last_path, present_path) in enumerate(data_loader):
        last_img, present_img = train_last_transforms.float().to(device), train_present_transforms.float().to(device)
        last_map, present_map = last_map.float().numpy(), present_map.float().numpy()

        prediction1, prediction2 = net(last_img, present_img)
        prediction1, prediction2 = prediction1.data.cpu().numpy(), prediction2.data.cpu().numpy()
        prediction1, prediction2 = np.squeeze(prediction1, axis=0), np.squeeze(prediction2, axis=0)
        last_map, present_map = np.squeeze(last_map, axis=0), np.squeeze(present_map, axis=0)
        gt_count1, gt_count2 = np.sum(last_map), np.sum(present_map)
        et_count1, et_count2 = np.sum(prediction1), np.sum(prediction2)
        mae += (abs(gt_count1 - et_count1) + abs(gt_count2 - et_count2))
        mse += ((gt_count1 - et_count1) * (gt_count1 - et_count1) + (gt_count2 - et_count2) * (gt_count2 - et_count2))
        if debug:
            last_path = '/'.join(last_path[0].split('/')[-1].split('\\')[-2:])
            present_path = '/'.join(present_path[0].split('/')[-1].split('\\')[-2:])
            last_dir, present_dir = osp.join("./test", last_path.split('/')[0][:-4]), \
                                    osp.join("./test", present_path.split('/')[0][:-4])
            last_path, present_path = last_path.split('/')[-1][:-4], present_path.split('/')[-1][:-4]
            if not os.path.exists(last_dir):
                os.mkdir(last_dir)
            if not os.path.exists(present_dir):
                os.mkdir(present_dir)
            utils.save_density_map(prediction1, last_dir, fname="{}_idx{}.png".format(last_path, idx))
            utils.save_density_map(prediction2, present_dir, fname="{}_idx{}.png".format(present_path, idx))
            utils.save_density_map(last_map, last_dir, fname="{}_idx{}_gt.png".format(last_path, idx))
            utils.save_density_map(present_map, present_dir, fname="{}_idx{}_gt.png".format(present_path, idx))
        temp = idx

    mae = mae/(2 * (temp + 1))
    mse = np.sqrt(mse / (2 * (temp + 1)))

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
