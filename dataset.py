from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img

import csv
import numpy as np
from PIL import Image
import cv2


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, path_csv_file):
        super(DatasetFromFolder, self).__init__()
        self.path_csv_file = path_csv_file
        self.direction = direction
        # self.a_path = join(image_dir, "data512x512_crop3")
        # self.b_path = join(image_dir, "data512x512_foreground3")
        # self.a_path = join(image_dir, "crop_p")
        # self.b_path = join(image_dir, "foreground_p")
        self.a_path = join(image_dir, "celeba_crop")
        self.b_path = join(image_dir, "celeba_full_parsing")
        self.image_filenames = []
        with open(self.path_csv_file, 'r') as f:
            reader = csv.reader(f)
            self.image_filenames = list(reader)
        # self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #
        # self.transform = transforms.Compose(transform_list)

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5)
            # , transforms.ColorJitter(0.5, 0, 0.5)   # brightness, contrast, saturation
        ])

    def __getitem__(self, index):
        finalsize = 512
        trans = random.randint(0, 10)   # random translate +-trans
        a = Image.open(join(self.a_path, self.image_filenames[index][0])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index][0])).convert('RGB')
        a = a.resize((finalsize+trans, finalsize+trans), Image.BICUBIC)
        b = b.resize((finalsize+trans, finalsize+trans), Image.BICUBIC)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        a = self.augmentation(a)
        random.seed(seed)
        torch.manual_seed(seed)
        b = self.augmentation(b)

        # arr_a = np.array(a)
        # gray_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2GRAY)
        # mask_a = np.zeros_like(gray_a)
        # mask_a[np.where(gray_a < 250)] = 255
        # mask_a = mask_a.astype(np.uint8)
        # mask_a = np.expand_dims(mask_a, axis=2)
        # mask_a = np.concatenate((mask_a, mask_a, mask_a), axis=2)
        #
        # arr_b = np.array(b)
        # gray_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2GRAY)
        # mask_b = np.zeros_like(gray_b)
        #
        # mask_b[np.where(gray_b < 250)] = 255
        # mask_b = mask_b.astype(np.uint8)
        # mask_b = np.expand_dims(mask_b, axis=2)
        # mask_b = np.concatenate((mask_b, mask_b, mask_b), axis=2)

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, finalsize+trans - finalsize - 1))
        h_offset = random.randint(0, max(0, finalsize+trans - finalsize - 1))
    
        a = a[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
        b = b[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]

        # mask_a = transforms.ToTensor()(mask_a)
        # mask_b = transforms.ToTensor()(mask_b)
        # mask_a = mask_a[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
        # mask_b = mask_b[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]


        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        # mask_a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(mask_a)
        # mask_b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(mask_b)


        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir, direction, path_csv_file):
        super(DatasetFromFolder2, self).__init__()
        self.path_csv_file = path_csv_file
        self.direction = direction
        self.a_path = join(image_dir, "crop_512x512")
        self.b_path = join(image_dir, "foreground_no_cloth")
        self.c_path = join(image_dir, "mask_512x512")
        self.image_filenames = []
        with open(self.path_csv_file, 'r') as f:
            reader = csv.reader(f)
            self.image_filenames = list(reader)

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5)
            # , transforms.ColorJitter(0.5, 0, 0.5)   # brightness, contrast, saturation
        ])

    def __getitem__(self, index):
        finalsize = 512
        trans = 50  # random translate +-trans
        a = Image.open(join(self.a_path, self.image_filenames[index][0])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index][0])).convert('RGB')
        c = Image.open(join(self.c_path, self.image_filenames[index][0])).convert('RGB')
        a = a.resize((finalsize + trans, finalsize + trans), Image.BICUBIC)
        b = b.resize((finalsize + trans, finalsize + trans), Image.BICUBIC)
        c = c.resize((finalsize + trans, finalsize + trans), Image.BICUBIC)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        a = self.augmentation(a)
        random.seed(seed)
        torch.manual_seed(seed)
        b = self.augmentation(b)
        random.seed(seed)
        torch.manual_seed(seed)
        c = self.augmentation(c)

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        c = transforms.ToTensor()(c)

        w_offset = random.randint(0, max(0, finalsize + trans - finalsize - 1))
        h_offset = random.randint(0, max(0, finalsize + trans - finalsize - 1))

        a = a[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
        b = b[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
        c = c[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if self.direction == "a2b":
            return a, b, c
        else:
            return b, a, c

    def __len__(self):
        return len(self.image_filenames)





if __name__ == "__main__":
    from data import get_training_set, get_training_set_with_mask
    from utils import save_img

    # root_path = "/home/zhang/zydDataset/faceRendererData/"
    root_path = "/home/zhang/zydDataset/asianFaces"
    train_set = get_training_set_with_mask(root_path, "a2b", './dataset/asian_train2.csv')
    a, b, c = train_set.__getitem__(0)
    c = c.numpy().transpose((1,2,0))*255
    c = c.astype(np.uint8)
    Image.fromarray(c).save("/home/zhang/zydDataset/asianFaces/c.png")
    # a, b = a.numpy(), b.numpy()
    # a, b = a.transpose((1,2,0)), b.transpose((1,2,0))
    # a = a*255
    # b = b*255
    # a = a.astype(np.uint8)
    # b = b.astype(np.uint8)
    # Image.fromarray(a).save("/home/zhang/zydDataset/faceRendererData/a.png")
    # Image.fromarray(b).save("/home/zhang/zydDataset/faceRendererData/b.png")
    save_img(a, "/home/zhang/zydDataset/asianFaces/a.png")
    save_img(b, "/home/zhang/zydDataset/asianFaces/b.png")


    print(1)