
# coding: utf-8

# In[1]:


from __future__ import print_function
from math import log10
from os import listdir
from os.path import join
import numpy as np
from PIL import Image


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


# In[3]:


import torch.utils.data as data
import torchvision.transforms as transforms


# ### Data loading funtions and classes

# In[4]:


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


# In[5]:


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.photo_path = join(image_dir, "a")
        self.sketch_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


# In[6]:


def get_dataset(data_path):
    return DatasetFromFolder(data_path)


# ### Define the networks
# 

# In[7]:


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# In[8]:


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), norm_layer(ndf * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                      padding=padw), norm_layer(ndf * nf_mult,
                                                affine=True), nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# In[9]:


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


# In[10]:


def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# ### Get train and test data

# In[11]:


train_set = get_dataset("./facades/train/")
test_set = get_dataset("./facades/test/")


# In[12]:


training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)


# In[13]:


input_nc = 3
output_nc = 3
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
batchSize = 1
lamb = 10
dataset = 'facades'


# In[14]:


netG = define_G(input_nc, output_nc, ngf, 'batch', False, [0])
netD = define_D(input_nc + output_nc, ndf, 'batch', False, [0])


# In[15]:


criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


# In[16]:


print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')


# In[17]:


real_a = torch.FloatTensor(batchSize, input_nc, 256, 256)
real_b = torch.FloatTensor(batchSize, output_nc, 256, 256)
real_a = Variable(real_a)
real_b = Variable(real_b)


#"""
netD = netD.cuda()
netG = netG.cuda()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()
criterionMSE = criterionMSE.cuda()
real_a = real_a.cuda()
real_b = real_b.cuda()
#"""
# ## Training module

# In[18]:


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        #print("iteration :",iteration)
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        #print("generating image :----------")
        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        #print("Feeding to the discriminator:---------")
        pred_fake = netD.forward(fake_ab.detach())
        #print("GANcriterion----------------")
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizerG.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))


# ### Test and Checkpts

# In[19]:


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        input = input.cuda()
        target = target.cuda()
        prediction = netG(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + dataset))


# In[1]:


for epoch in range(1, 201):
    train(epoch)
    test()
    if epoch % 100 == 0:
        checkpoint(epoch)


