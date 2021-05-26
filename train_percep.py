from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from losses import *

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=28, help='the starting epoch count')  # 10
parser.add_argument('--continue_train', type=bool, default=True, help='wheather to continue')  # 10
parser.add_argument('--n_epochs', type=int, default=100, help='# of iter at starting learning rate')    # 100
parser.add_argument('--n_epochs_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')   # 100
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')   # 0.0002
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=2, help='multiply by a gamma every lr_decay_iters iterations')  # 50  step方式用得到
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--percep', type=int, default=20, help='weight on L1 term in objective')
opt = parser.parse_args()

# n_epochs的含义是保持最初的学习率的那些epoch，n_epochs_decay的含义是学习率发生下降的那些epoch
# 比如有50个epoch，那么可能n_epochs=25, n_epochs_decay=25，则25个epoch为初始学习率，后面25个才发生学习率下降
# 所以总的epoch数是 n_epochs + n_epochs_decay

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
# root_path = "/home/zhang/zydDataset/faceRendererData/"
# train_set = get_training_set(root_path, opt.direction, './dataset/celeba_train_crop3.csv')
# root_path = "/home/yexiaoqi/PycharmPro/zyd/asianFaces_parsing/"
root_path = "/home/zhang/zydDataset/faceRendererData/trainData/train_data_pix2pix_hair/"
train_set = get_training_set(root_path, "a2b", './dataset/Celeba_and_stylegan.csv')
# test_set = get_test_set(root_path, opt.direction, './dataset/asian_test5.csv')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_512', 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
if opt.continue_train:
    net_g = torch.load("checkpoint_0314/netG_model_epoch_latest.pth")
    net_d = torch.load("checkpoint_0314/netD_model_epoch_latest.pth")

def convert_img(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
# criterionEdge = EdgeLoss().to(device)
criterionVGG = VGGLoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

losslist_g = []
losslist_d = []
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)
        # save_img(real_b[0].detach().cpu(), "./input.jpg")
        ######################
        # (1) Update D network
        ######################

        # save image while training
        if iteration % 100 == 0:
            a = real_a.detach().cpu()[0, :, :, :]
            a = convert_img(a)
            # a = postprocessing(a)
            Image.fromarray(a).save("checkpoint_0314/real_a_{}.png".format(epoch))

            b = real_b.detach().cpu()[0, :, :, :]
            b = convert_img(b)
            # b = postprocessing(b)
            Image.fromarray(b).save("checkpoint_0314/real_b_{}.png".format(epoch))

            c = fake_b.detach().cpu()[0, :, :, :]
            c = convert_img(c)
            # c = postprocessing(c)
            Image.fromarray(c).save("checkpoint_0314/fake_b_{}.png".format(epoch))

        #=========================

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        # loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g_l1 = criterionL1(fake_b, real_b) * 10
        # loss_g_percep = criterionPercep(fake_b, real_b) * opt.percep
        # loss_g_edge = criterionEdge(fake_b, real_b) * 50

        # VGG feature matching loss
        loss_g_VGG = criterionVGG(fake_b, real_b) * 10

        # feature matching loss
        # loss_g_feat = 0
        # for j in range(len(pred_fake[0])-1):
        #     loss_g_feat += criterionL1(pred_fake[0][j], pred_real[0][j].detach())*100

        # loss_g = loss_g_gan + loss_g_l1  #  + loss_g_edge  # + loss_g_percep
        # loss_g = loss_g_gan + loss_g_VGG + loss_g_feat
        # loss_g = loss_g_gan + loss_g_l1 + loss_g_feat

        # loss_g = loss_g_gan.item() + loss_g_l1.item() + loss_g_VGG.item()
        loss_g = loss_g_gan + loss_g_l1 + loss_g_VGG
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        losslist_g.append(loss_g.item())
        losslist_d.append(loss_d.item())


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    # avg_psnr = 0
    # for batch in testing_data_loader:
    #     input, target = batch[0].to(device), batch[1].to(device)
    #
    #     prediction = net_g(input)
    #     mse = criterionMSE(prediction, target)
    #     psnr = 10 * log10(1 / mse.item())
    #     avg_psnr += psnr
    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 20 == 0:
        if not os.path.exists("checkpoint_0314"):
            os.mkdir("checkpoint_0314")
        # if not os.path.exists("./checkpoint"):
        #     os.mkdir("./checkpoint")
        net_g_model_out_path = "checkpoint_0314/netG_model_epoch_{}.pth".format(epoch)
        net_d_model_out_path = "checkpoint_0314/netD_model_epoch_{}.pth".format(epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to ./checkpoint")

    if not os.path.exists("checkpoint_0314"):
        os.mkdir("checkpoint_0314")

    torch.save(net_g, "checkpoint_0314/netG_model_epoch_latest.pth")
    torch.save(net_d, "checkpoint_0314/netD_model_epoch_latest.pth")

    np.savetxt("checkpoint_0314/loss_g.txt", np.array(losslist_g), fmt="%.4f")
    np.savetxt("checkpoint_0314/loss_d.txt", np.array(losslist_d), fmt="%.4f")
    np.savetxt("checkpoint_0314/epoch.txt", np.array([epoch]), fmt="%d")


print("End of training")


