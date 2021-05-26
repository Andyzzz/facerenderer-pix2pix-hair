from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

import csv
import cv2
import numpy as np
from PIL import Image
import glob
from utils import *


def writecsv(image_dir, csv_name):
    lis = glob.glob(image_dir + "*.png")
    if not lis:
        lis = glob.glob(image_dir + "*.jpg")
    f0 = open(csv_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(f0)
    for i in range(0, len(lis)):
        name = lis[i].split("/")[-1]
        csv_writer.writerow([name])
    f0.close()


def get_img(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


def findkey(im, i, j, color_value_dict):
    r0 = im[i, j, 0]
    g0 = im[i, j, 1]
    b0 = im[i, j, 2]
    g_r0 = float(g0+1)/float(r0+1)
    b_r0 = float(b0+1)/float(r0+1)

    res = [255, 0, 0]
    diff = 1e10
    for key in color_value_dict.keys():
        diff_r = abs(r0 - key[0])
        diff_g = abs(g0 - key[1])
        diff_b = abs(b0 - key[2])
        if diff_r < 30 and diff_g < 30 and diff_b < 30:
            im[i, j, :] = np.array(list(key))[:]
            return list(key)

    for m in range(i-1, 0, -1):
        if tuple(im[m, j, :]) in color_value_dict.keys():
            im[i, j, :] = im[m, j, :]
            res = [im[m, j, 0], im[m, j, 1], im[m, j, 2]]
            break

    return res


def postprocessing(im):
    color_value_dict = {(0, 0, 0): 0,
                        (0, 0, 255): 1, (255, 0, 0): 2,
                        (150, 30, 150): 3, (255, 65, 255): 4,
                        (150, 80, 0): 5, (170, 120, 65): 6,
                        (125, 125, 125): 7, (255, 255, 0): 8,
                        (0, 255, 255): 9, (255, 150, 0): 10, (255, 225, 120): 11,
                        (255, 125, 125): 12, (200, 100, 100): 13, (0, 255, 0): 14,
                        (0, 150, 80): 15, (215, 175, 125): 16, (220, 180, 210): 17, (125, 125, 255): 18}
    h, w, c = im.shape
    for i in range(h):
        for j in range(w):
            a = tuple(im[i, j, :])
            if a not in color_value_dict.keys():
                v = np.array(findkey(im, i, j, color_value_dict))
                im[i, j, :] = v[:]

    return im


def normalize_SEAN(img):
    # 相当于把图像放大1.1倍后再向下移动２０像素
    scale = 1.1
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    res = []

    if len(img.shape) == 2:
        res = np.zeros((512, 512), dtype=np.uint8)
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :] = img[top:top + 512, left:left + 512]

    elif len(img.shape) == 3 and img.shape[2] == 3:
        res = np.ones((512, 512, 3), dtype=np.uint8) * 255
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :, :] = img[top:top + 512, left:left + 512, :]

    return res


def load_img(filepath, normalize_flag=True):
    img = Image.open(filepath).convert('RGB')
    if normalize_flag:
        img2 = normalize_SEAN(np.array(img))
        img = Image.fromarray(img2)

    return img


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=180, help='saved model of which epochs')  # 180
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

# 最初始的版本是checkpoint_parsing里的epoch_180，输出会多一圈
# 在青云电脑上重新训练的版本是checkpoint_align中的epoch_140，增加了perceptual loss，并且输出不会多一圈
# 0314加入stylegan生成的侧脸图像，重新训练模型，保存在checkpoint_parsing_0314里,epoch是60

# model_path = "checkpoint_align/netG_model_epoch_{}.pth".format(opt.nepochs)   # checkpoint_parsing   align
model_path = "checkpoint_parsing/netG_model_epoch_{}.pth".format(opt.nepochs)

net_g = torch.load(model_path).to(device)

# image_dir = "/home/yexiaoqi/PycharmPro/zyd/rawscan_parsing/"
#
# # csv读取文件
# test_csv_file = "./dataset/rawscan_test_parsing.csv"
# if not os.path.isfile(test_csv_file):
#     writecsv(image_dir, test_csv_file)
#     print("write csv file ended ")
#
# image_filenames = []
# with open(test_csv_file, 'r') as f1:
#     reader = csv.reader(f1)
#     image_filenames = list(reader)

# image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

concat = False
count = 0
# image_dir = "/home/zhang/zydDataset/faceRendererData/TUFront_parsing_crop/"
# image_filenames = sorted(glob.glob(image_dir + "*_*/*.png"))


# ===========================================================================================
# image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/0324_addhair/"
# tgt_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0324_addhair_3/"
# image_filenames = sorted(glob.glob(image_dir + "*.png"))
image_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/1_face-parsing/0517/addhair/"
image_filenames = sorted(glob.glob(image_dir + "*.png"))
tgt_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/2_facerender-pix2pix-hair/0517_addhair/"

for i in range(0, len(image_filenames)):
    image_path = image_filenames[i]
    image_name = image_path.split("/")[-1]

    raw_img = load_img(image_path, False)
    # 为了测试侧脸图像加入的平移
    # raw_img = raw_img.rotate(0, expand=1, fillcolor=(0, 0, 0), translate=(-60, 0))

    img = transform(raw_img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    out_img = get_img(out_img)

    out_img = postprocessing(out_img)

    out_img = align_crop_full(out_img, np.array(raw_img))

    if concat:
        concat_img = cv2.hconcat([np.array(raw_img), out_img])
    else:
        concat_img = out_img

    # subdir = image_path.split("/")[-2]
    # tgt_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0125/"
    # if not os.path.exists(tgt_dir + subdir):
    #     os.makedirs(tgt_dir + subdir)
    # Image.fromarray(concat_img).save(os.path.join(tgt_dir + subdir, image_name))


    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    Image.fromarray(concat_img).save(os.path.join(tgt_dir, image_name))

    count += 1

    print(i, image_name)
# ========================================================================================



# # 0311版本，带不同pose
# image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/crop/"
# image_filenames = sorted(glob.glob(image_dir + "*/*.png"))   # 0311
# for i in range(0, len(image_filenames)):
#     image_path = image_filenames[i]
#     image_name = image_path.split("/")[-1]
#     pose_ind = image_path.split("/")[-2]  # 0311
#     raw_img = load_img(image_path, False)
#     img = transform(raw_img)
#     input = img.unsqueeze(0).to(device)
#     out = net_g(input)
#     out_img = out.detach().squeeze(0).cpu()
#
#     out_img = get_img(out_img)
#
#     out_img = postprocessing(out_img)
#
#     out_img = align_crop_full(out_img, np.array(raw_img))
#
#     if concat:
#         concat_img = cv2.hconcat([np.array(raw_img), out_img])
#     else:
#         concat_img = out_img
#
#     tgt_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0325/" + pose_ind + "/"
#     if not os.path.exists(tgt_dir):
#         os.makedirs(tgt_dir)
#     Image.fromarray(concat_img).save(os.path.join(tgt_dir, image_name))
#
#     count += 1
#     print(i, image_name)

