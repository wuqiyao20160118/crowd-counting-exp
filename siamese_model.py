# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import collections
import h5py
import numpy as np


class CSRNet(nn.Module):
    def __init__(self, load_weights=False, layer3=False):
        super(CSRNet, self).__init__()
        self.layer3 = layer3
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat2 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.frontend = make_layers(self.frontend_feat)
        self.frontend2 = make_layers(self.frontend_feat2)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
#            print("VGG",list(mod.state_dict().items())[0][1])#要的VGG值
            fsd = collections.OrderedDict()
            fsd2 = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):  # 10个卷积*（weight，bias）=20个参数
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

            for i in range(len(self.frontend2.state_dict().items())):  # 10个卷积*（weight，bias）=20个参数
                temp_key = list(self.frontend2.state_dict().items())[i][0]
                fsd2[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend2.load_state_dict(fsd2)

    def forward(self, x):
        out = self.frontend(x)
        if not self.layer3:
            return out
        else:
            y = self.frontend2(x)
            return out, y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, d_rate=None):
    if dilation:
        if d_rate is None:
            d_rate = 2
        else:
            d_rate = d_rate
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CoattentionModel(nn.Module):
    def __init__(self, all_channel=256, all_dim=59*59, output_size=473, output=True):	 #473./8=60
        super(CoattentionModel, self).__init__()
        self.output_size = output_size
        self.output = output

        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.prelu = nn.ReLU(inplace=True)

        self.scale1 = make_layers([128], in_channels=512, dilation=True)
        self.scale2 = make_layers([128], in_channels=512, dilation=True, d_rate=4)
        self.scale3 = make_layers([128], in_channels=512, dilation=True, d_rate=8)
        self.scale4 = make_layers([128], in_channels=512, dilation=True, d_rate=12)

        # [512, 512, 512, 256, 128, 64]
        #self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend_feat = [256, 256, 256, 128, 64]
        self.backend_feat_scale = [512, 512, 512, 256]
        self.backend_feat1 = [512, 512, 512, 256]
        self.backend_feat2 = [256, 128, 64]
        self.backend1 = make_layers(self.backend_feat1, in_channels=512, dilation=True)
        self.backend2 = make_layers(self.backend_feat2, in_channels=256*3, dilation=True)
        self.backend = make_layers(self.backend_feat_scale, in_channels=512, dilation=True)

        self.main_classifier1 = nn.Conv2d(64, 1, kernel_size=1)
        self.main_classifier2 = nn.Conv2d(64, 1, kernel_size=1)

        #self.seg_backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.seg_backend = make_layers(self.backend_feat, in_channels=256, dilation=True)
        self.seg_output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.encoder = CSRNet(layer3=True)
        #self.encoder = CSRNet(layer3=False)
        for k, v in self.encoder.named_parameters():
            v.requires_grad = False
        for k, v in self.seg_backend.named_parameters():
            v.requires_grad = False
        for k, v in self.seg_output_layer.named_parameters():
            v.requires_grad = False

        h5f = h5py.File("./weight/CSR_scale_seg_layer3_scvd_38.h5", mode='r')
        #h5f = h5py.File("./weight/CSR_scale_seg_whole_scvd_108.h5", mode='r')
        for k, v in self.seg_backend.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.backend."+k]))
            v.copy_(param)
        for k, v in self.seg_output_layer.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.output_layer." + k]))
            v.copy_(param)

        h5f = h5py.File("./weight/CSR_scale_scvd_4branch_120.h5", mode='r')
        for k, v in self.backend.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.backend."+k]))
            v.copy_(param)
        for k, v in self.scale1.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.scale1." + k]))
            v.copy_(param)
        for k, v in self.scale2.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.scale2." + k]))
            v.copy_(param)
        for k, v in self.scale3.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.scale3." + k]))
            v.copy_(param)
        for k, v in self.scale4.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f["DME.scale4." + k]))
            v.copy_(param)

    def learnable_parameters(self, lr):
        parameters = [
            {'params': self.encoder.parameters(), 'lr': 0},
            {'params': self.seg_backend.parameters(), 'lr': 0},
            {'params': self.seg_output_layer.parameters(), 'lr': 0},
            {'params': self.linear_e.parameters(), 'lr': lr},
            {'params': self.gate.parameters(), 'lr': lr},
            {'params': self.scale1.parameters(), 'lr': lr},
            {'params': self.scale2.parameters(), 'lr': lr},
            {'params': self.scale3.parameters(), 'lr': lr},
            {'params': self.scale4.parameters(), 'lr': lr},
            {'params': self.backend.parameters(), 'lr': lr},
            {'params': self.backend1.parameters(), 'lr': lr},
            {'params': self.backend2.parameters(), 'lr': 10 * lr},
            {'params': self.main_classifier1.parameters(), 'lr': 10 * lr},
            {'params': self.main_classifier2.parameters(), 'lr': 10 * lr}
        ]
        return parameters

    def forward(self, input1, input2):
        #exemplar = self.encoder(input1)
        #query = self.encoder(input2)
        exemplar, exemplar_seg_pre = self.encoder(input1)
        query, query_seg_pre = self.encoder(input2)

        exemplar1 = self.scale1(exemplar)
        query1 = self.scale1(query)
        exemplar2 = self.scale2(exemplar)
        query2 = self.scale2(query)
        exemplar3 = self.scale3(exemplar)
        query3 = self.scale3(query)
        exemplar4 = self.scale4(exemplar)
        query4 = self.scale4(query)
        exemplar_scale = torch.cat([exemplar1, exemplar2, exemplar3, exemplar4], 1)
        query_scale = torch.cat([query1, query2, query3, query4], 1)

        exemplar_scale = self.backend(exemplar_scale)
        query_scale = self.backend(query_scale)

        #exemplar_seg = self.seg_output_layer(self.seg_backend(exemplar))
        #query_seg = self.seg_output_layer(self.seg_backend(query))
        exemplar_seg = self.seg_output_layer(self.seg_backend(exemplar_seg_pre))
        query_seg = self.seg_output_layer(self.seg_backend(query_seg_pre))
        exemplar_seg = self.sigmoid(exemplar_seg)
        query_seg = self.sigmoid(query_seg)

        exemplar = self.backend1(exemplar)
        query = self.backend1(query)
        fea_size = query.size()[2:]

        exemplar_flat = exemplar.view(-1, self.channel, self.dim)  # N,C,H*W
        query_flat = query.view(-1, self.channel, self.dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)
        A = F.softmax(A, dim=1)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        #exemplar = exemplar * exemplar_seg
        #query = query * query_seg
        exemplar = exemplar_seg_pre * exemplar_seg
        query = query_seg_pre * query_seg
        input1_att = F.interpolate(input1_att, scale_factor=2, mode="bilinear")
        input2_att = F.interpolate(input2_att, scale_factor=2, mode="bilinear")
        exemplar_scale = F.interpolate(exemplar_scale, scale_factor=2, mode="bilinear")
        query_scale = F.interpolate(query_scale, scale_factor=2, mode="bilinear")

        input1_att = torch.cat([input1_att, exemplar, exemplar_scale], 1)
        input2_att = torch.cat([input2_att, query, query_scale], 1)

        input1_att = self.backend2(input1_att)
        input2_att = self.backend2(input2_att)
        if self.output:
            input1_att = self.main_classifier1(input1_att)
            input2_att = self.main_classifier2(input2_att)

        return input1_att, input2_att


class CoattentionScale(nn.Module):
    def __init__(self, weight_dir=None):
        super(CoattentionScale, self).__init__()
        self.CAM_1 = nn.Conv2d(64, 64, kernel_size=3)
        self.CAM_2 = nn.Conv2d(64, 32, kernel_size=3)
        self.CAM_3 = nn.Conv2d(32, 3, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()
        self.CountNet = CoattentionModel(output=False)
        if weight_dir:
            pretrained_dict = torch.load(weight_dir, map_location="cpu")['model']
            model_dict = self.CountNet.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "main_classifier" not in k}
            model_dict.update(pretrained_dict)
            self.CountNet.load_state_dict(model_dict)

        for k, v in self.CountNet.named_parameters():
            v.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1, x2 = self.CountNet(x1, x2)
        x = self.relu(self.CAM_1(x1))
        x = self.relu(self.CAM_2(x))
        x = self.relu(self.CAM_3(x))
        return x

    def learnable_parameters(self, lr):
        parameters = [
            {'params': self.CountNet.parameters(), 'lr': 0},
            {'params': self.CAM_1.parameters(), 'lr': lr},
            {'params': self.CAM_2.parameters(), 'lr': lr},
            {'params': self.CAM_3.parameters(), 'lr': lr},
        ]
        return parameters


def CoattentionNet():
    model = CoattentionModel()
    return model


def CoattentionNet_flow():
    model = CoattentionModel()
    del model.final_conv2
    del model.final_conv1
    return model


if __name__ == "__main__":
    model = CoattentionModel()
    #model_dict = model.state_dict()
    #model.train()
    #for k, v in model.named_parameters():
        #print(k, v.requires_grad)
    model.eval()
    x = torch.rand([1, 3, 473, 473])
    y = torch.rand([1, 3, 473, 473])
    i1, i2 = model(x, y)
    print(i1.size())
