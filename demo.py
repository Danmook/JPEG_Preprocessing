import torch
from model_list import model_list
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='preprocess', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints/exp4_350000', type=str)
    parser.add_argument('--save_path', default='./demo', type=str)
    parser.add_argument('--demo_path', default=r'kodim03_9.png', type=str)
    a = parser.parse_args()

    model = model_list(a, 0).to(device)
    state_dict_com = load_checkpoint(a.checkpoint_path, device)
    model.load_state_dict(state_dict_com['model'])
    model.eval()

    image = Image.open(a.demo_path).convert('RGB')
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    inv_transform = transforms.ToPILImage()

    img = transform(image)
    img_ = 255 * img.unsqueeze(0).to(device)
    _, results = model(img_)
    img_pre = inv_transform(results[0].squeeze(0))
    img_edge = inv_transform(results[1].squeeze(0))
    img_ae = inv_transform(results[2].squeeze(0))
    print(torch.mean((img - results[0].squeeze(0))**2))
    # print(results[2].squeeze(0))
    img_pre.show("img_pre")
    img_edge.show("img_edge")
    img_ae.show("img_ae")