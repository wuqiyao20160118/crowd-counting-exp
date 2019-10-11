# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv= nn.Conv2d(2048, depth, 1,1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0], dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1], dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2], dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(depth*5, 256, kernel_size=3, padding=1)  #512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        #for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)#classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)
        

    def forward(self, x):
        #out = self.conv2d_list[0](x)
        #mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size=x.shape[2:]
        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features=F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0) 
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1) 
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2) 
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3) 
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        #for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)
        
        return out
  


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ASPP, [6, 12, 18], [6, 12, 18], 512)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        #input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fea = self.layer5(x)
        return fea


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
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
    def __init__(self, block, layers, all_channel=256, all_dim=60*60, output_size=473):	 #473./8=60
        super(CoattentionModel, self).__init__()
        self.output_size = output_size
        self.encoder = ResNet(block, layers)
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.prelu = nn.ReLU(inplace=True)
        # [512, 512, 512, 256, 128, 64]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend1 = make_layers(self.backend_feat, in_channels=all_channel, dilation=True)
        self.backend2 = make_layers(self.backend_feat, in_channels=all_channel, dilation=True)
        self.final_layer1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(inplace=True))
        self.final_layer2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for k, v in self.encoder.named_parameters():
            if "layer5" not in k:
                v.requires_grad = False

    def learnable_parameters(self, lr):
        parameters = [
            {'params': self.encoder.conv1.parameters(), 'lr': 0},
            {'params': self.encoder.bn1.parameters(), 'lr': 0},
            {'params': self.encoder.layer1.parameters(), 'lr': 0},
            {'params': self.encoder.layer2.parameters(), 'lr': 0},
            {'params': self.encoder.layer3.parameters(), 'lr': 0},
            {'params': self.encoder.layer4.parameters(), 'lr': 0},
            {'params': self.encoder.layer5.parameters(), 'lr': lr},
            {'params': self.linear_e.parameters(), 'lr': lr},
            {'params': self.gate.parameters(), 'lr': lr},
            {'params': self.conv1.parameters(), 'lr': lr},
            {'params': self.conv2.parameters(), 'lr': lr},
            {'params': self.backend1.parameters(), 'lr': 10 * lr},
            {'params': self.backend2.parameters(), 'lr': 10 * lr},
            {'params': self.final_layer1.parameters(), 'lr': 10 * lr},
            {'params': self.final_layer2.parameters(), 'lr': 10 * lr}
        ]
        return parameters

    def forward(self, input1, input2):
        exemplar = self.encoder(input1)
        query = self.encoder(input2)
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
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = F.relu(input1_att, inplace=True)
        input2_att = F.relu(input2_att, inplace=True)
        input1_att = self.backend1(input1_att)
        input2_att = self.backend2(input2_att)
        input1_att = self.final_layer1(input1_att)
        input2_att = self.final_layer2(input2_att)

        input1_att = F.interpolate(input1_att, self.output_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        input2_att = F.interpolate(input2_att, self.output_size, mode='bilinear', align_corners=False)

        return input1_att, input2_att
    

def Res_Deeplab():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def CoattentionNet(training=True):
    model = CoattentionModel(Bottleneck, [3, 4, 23, 3])
    if training:
        pretrained_dict = torch.load("co_attention.pth")['model']
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if "encoder" in k and "main_classifier" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        #for k, v in model.named_parameters():
        #   print(k, v.requires_grad)
    return model


def CoattentionNet_flow():
    model = CoattentionModel(Bottleneck, [3, 4, 23, 3])
    del model.final_conv2
    del model.final_conv1
    return model


if __name__ == "__main__":
    model = CoattentionNet()
    #model.eval()
    #x = torch.rand([1, 3, 473, 473])
    #y = torch.rand([1, 3, 473, 473])
    #i1, i2 = model(x, y)
    #print(i1.size())
