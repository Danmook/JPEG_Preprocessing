import torch
import torch.nn as nn
from modules import *
from utils import freeze
import torchvision
class Preprocess(nn.Module):
    """
    图像预处理部分,包括encoder、decoder以及RCF边缘检测,不包含模拟JPEG编解码部分
    """
    def __init__(self, a, rank) -> None:
        super(Preprocess, self).__init__()
        self.a = a
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_net(192)
        self.decoder = Synthesis_net(192)
        # self.ae = 
        self.edge = RCF(rank)
        self.edge.load_state_dict(torch.load('bsds500_pascal_model.pth'))
        freeze(self.edge)
        # self.edge = self.edge.to(self.device)

    def forward(self, inputs):
        """
        :param inputs: mini-batch
        :return img_pre: 预处理后的图像 img_edge: 边缘 img_ae:自编码器填充
        """
        # edge = RCF()
        # edge.load_state_dict('bsds500_pascal_model.pth')
        # 前向传播过程
        img_edge = self.edge(255*inputs)[5]
        # inputs_con = torch.cat((inputs, img_edge), dim=1)
        img_edge = img_edge.repeat(1, 3, 1, 1)  
        feature = self.encoder(inputs)
        if self.training:
            feature_hat = self._noisy(feature, is_train=True)
        else:
            feature_hat = feature
        img_ae = torch.clamp(self.decoder(feature_hat), 0, 1)
        # img_pre = torch.sigmoid(img_ae + img_edge)
        img_pre = img_ae
        results = [img_pre, img_edge, img_ae]
        img_ae = self.decoder(self.encoder(inputs))
        img_edge = self.edge(255*inputs)[5]
        img_pre = torch.sigmoid(img_ae + img_edge)
        # Loss
        # MSE
        distortion = torch.mean((inputs - img_edge)**2)
        loss = distortion

        return loss, results

    def loss(self, inputs, img_pre):
        pass

    def _noisy(self, f, is_train=False):
        if is_train:
            uniform_noise = nn.init.uniform_(torch.zeros_like(f), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            f_hat = f + uniform_noise
        else:
            f_hat = f
        return f_hat
