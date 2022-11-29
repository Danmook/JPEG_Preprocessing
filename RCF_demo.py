import argparse
import torch
from models import RCF
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os

def multi_scale(model, path, save_dir):
    image = cv2.imread(path)
    # print(image.shape)
    H, W, _ = image.shape
    ms_fuse = np.zeros((H, W), np.float32)
    scale = [0.5, 1, 1.5]
    for k in range(len(scale)):
        im_ = cv2.resize(image, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
        im_ = im_.transpose((2, 0, 1)).astype(np.float32)
        input = torch.from_numpy(im_).unsqueeze(0)
        # print(type(input))
        results = model(input)
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse_res
    ms_fuse = ms_fuse / len(scale)
    #归一化
    ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
    ms_fuse = ((ms_fuse) * 255).astype(np.uint8)
    # fuse_res = (fuse_res - fuse_res.min()) / (fuse_res.max() - fuse_res.min())
    # fuse_res = ((fuse_res) * 255).astype(np.uint8)
    filename = "muti_edge.png"
    cv2.imwrite(os.path.join(save_dir, filename), ms_fuse)
    print('Running multi-scale test done')


def single_scale(model, path, save_dir):
    image = cv2.imread(path)
    H, W, _ = image.shape
    im_ = image.transpose((2, 0, 1)).astype(np.float32)
    input = torch.from_numpy(im_).unsqueeze(0)
    results = model(input)
    for i in range(len(results)):
        fuse_res = torch.squeeze(results[i].detach()).cpu().numpy()
        # fuse_res = (fuse_res - fuse_res.min()) / (fuse_res.max() - fuse_res.min())
        fuse_res = ((fuse_res) * 255).astype(np.uint8)
        if i==len(results)-1:
            filename = "single_edge.png"
        else:
            filename = "single_s{}.png".format(i+1)
        cv2.imwrite(os.path.join(save_dir, filename), fuse_res)
    print('Running single-scale test done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEMO')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--checkpoint', default='bsds500_pascal_model.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--path', help='path to the image', default='kodim03.png')

    args = parser.parse_args()
    # 实例化网络
    model = RCF()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    save_dir = args.path.split('.')[0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    multi_scale(model, args.path, save_dir)
    single_scale(model, args.path, save_dir)
