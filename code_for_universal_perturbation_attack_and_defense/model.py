import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
# import torchvision.models as models
import torchvision.models  # IMAGENET
import models  # CIFAR 10
import tqdm
import PIL
from pytorch_unet import UNet

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])
CIFAR10_SIZE = 32

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_SIZE = 224


MEAN = None
STD = None
SIZE = None


# CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = None
def get_conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = get_conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = get_conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PertFilteringNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10, n_res_blocks=5):
        super(PertFilteringNet, self).__init__()
        self.conv1 = get_conv3x3(in_channels, 64)
        # TODO: is it ok to use list here
        self.resblocks = nn.ModuleList([ResidualBlock(64, 64) for _ in range(n_res_blocks)])
        self.conv12 = get_conv3x3(64, 16)
        self.output = get_conv3x3(16, 3)

    def forward(self, x):
        x = self.conv1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.conv12(x)
        x = self.output(x)
        return x


class JoinedNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10, pretrained_model=None, n_res_blocks=5, use_unet=False):
        super(JoinedNet, self).__init__()
        # FIXME: do we need two pretrained models here??
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        if use_unet:
            self.pfn = UNet(in_channels)
        else:
            self.pfn = PertFilteringNet(in_channels=in_channels, n_classes=n_classes, n_res_blocks=n_res_blocks)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_clean, x_pert):
        self.pretrained_model.eval()
        # We expect the x to be transformed (normalized).
        # First compute original classifer with clean input
        y = self.pretrained_model(x_clean)
        # Then compute filtered classifer with dirty input
        # 1. Filter
        x_filtered = self.pfn(x_pert)
        # 2. Classify
        y_hat = self.pretrained_model(x_filtered)
        return y, y_hat, x_filtered



def load_data(datafolder, batch_size=64, aug=True, shuffle=True):
    # For CIFAR10
    if aug:
        trans = [
            torchvision.transforms.Resize((SIZE, SIZE)), 
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            torchvision.transforms.RandomCrop(SIZE, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    else:
        trans = [
            torchvision.transforms.Resize((SIZE, SIZE)), 
        ]
    train_transforms = torchvision.transforms.Compose(trans + [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEAN, STD)
    ])
    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((224,224)), 
    #     torchvision.transforms.ToTensor(), 
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    data = torchvision.datasets.ImageFolder(
        root=datafolder,
        transform=train_transforms
    )

    # data = torchvision.datasets.CIFAR10(
    #     './data_cifar10', download=True, train=True, transform=train_transforms)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    print('Finish Loading Data')
    return data_loader


def build_loss(model: JoinedNet, learning_rate=0.01):
    error = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    return error, optimizer


def load_pert(folder):
    pert_list = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if file_path.endswith('.npy'):
            pert = np.load(file_path)
            if len(pert.shape) == 4:
                pert = pert[0]
            pert_list.append(torch.tensor(pert))
    return pert_list


def unnormalize(img):
    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape))
        A = torch.min(A, maxv*torch.ones(A.shape))
        return A
    clip = lambda x: clip_tensor(x, 0, 1.0)
    img_trans_back = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0, 0, 0], std=1.0/STD),
        torchvision.transforms.Normalize(mean=-MEAN, std=[1, 1, 1]),
        torchvision.transforms.Lambda(clip),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(SIZE)])
    return img_trans_back(img)


def idx_to_label(idx: int, dataset='cifar10'):
    def init_classes(labels_path: str):
        with open(labels_path, 'r') as f:
            return [ name for name in f.read().split('\n')]

    global classes
    if dataset.lower() == 'cifar10':
        labels_path = 'cifar10.txt'
    elif dataset.lower() == 'imagenet':
        labels_path = 'synset_words.txt'
    elif dataset.lower() == 'natural_images':
        labels_path = 'natural_images.txt'
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    if classes is None:
        classes = init_classes(labels_path)
    
    return classes[idx]



def predict(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    model:
    img: HWC
    return: index of most possible class
    """
    logits = model(img)
    prob = torch.max(torch.softmax(logits, 1)).item()
    return logits.argmax(1), prob


def save_all(folder, net, dataloader, pert_list, device, dataset=''):
    os.makedirs(folder, exist_ok=True)
    pert = random.choice(pert_list)
    i = 0
    types = ('orig', 'pert', 'filtered')
    for imgs, _ in dataloader:
        for img in imgs:
            # Get label
            img_tuple, label_tuple, prob_tuple = compare(net, img, pert, device, dataset=dataset)
            for img, type, label, prob in zip(img_tuple, types, label_tuple, prob_tuple):
                path = os.path.join(folder, '{:03}_{}_predicted_{}_confidence_{}.png'.format(i, type, label, prob))
                img.save(path)
            i += 1


def compare(model: JoinedNet, img, pert, device, show=False, dataset=''):
    # img: [3, H, W]
    img_pert = (img[None, :] + pert).to(device)
    img_filtered = filter_img(model, img_pert)[0]
    # Predict
    y, y_prob = predict(model.pretrained_model, img[None, :].to(device))
    y = idx_to_label(y, dataset=dataset)
    y_pert, y_pert_prob = predict(model.pretrained_model, img_pert)
    y_pert = idx_to_label(y_pert, dataset=dataset)
    y_filtered, y_filtered_prob = predict(model.pretrained_model, img_filtered[None, :].to(device))
    y_filtered = idx_to_label(y_filtered, dataset=dataset)
    # Recover original image
    img = unnormalize(img)
    img_pert = unnormalize(img_pert[0].cpu())
    img_filtered = unnormalize(img_filtered.cpu())
    if show:
        img.show()
        img_pert.show()
        img_filtered.show()
    return (img, img_pert, img_filtered), (y, y_pert, y_filtered), (y_prob, y_pert_prob, y_filtered_prob)


def compute_fool_number(pretrained_model, imgs_orig, imgs_pert):
    y = torch.argmax(pretrained_model(imgs_orig), 1)
    y_hat = torch.argmax(pretrained_model(imgs_pert), 1)
    # print('y:')
    # print(y)
    # print('y_hat:')
    # print(y_hat)
    return torch.sum(y != y_hat).item()


def compare_fool_rate(net: JoinedNet, pretrained_model, dataloader, pert_list, device):
    net.eval()
    n = len(dataloader.dataset)
    n_fooled = 0
    n_fooled_filtered = 0
    # i = 0
    for imgs, _ in dataloader:
        pert = random.choice(pert_list)
        imgs_pert = (imgs + pert).to(device)
        imgs = imgs.to(device)
        imgs_filtered = net.pfn(imgs_pert)
        n_fooled += compute_fool_number(pretrained_model, imgs, imgs_pert)
        n_fooled_filtered += compute_fool_number(pretrained_model, imgs, imgs_filtered)
        # # TODO: remove me
        # i += 1
        # if i > 1:
        #     break
    print('Fooling rate (before): {:.5f}'.format(1.0 * n_fooled / n))
    print('Fooling rate (after): {:.5f}'.format(1.0 * n_fooled_filtered / n))


def filter_img(model: JoinedNet, img):
    model.eval()
    filtered = model.pfn(img)
    return filtered


def train_pfn(
        debug=False, epochs=10, print_every=1, pretrained_model='resnet18', n_res_blocks=5,
        batch_size=4, lr=0.05, use_unet=False, use_aug=True, recover_loss_weight=20, label_loss_weight=1.0, dataset='', datafolder='', pertfolder='', need_save=False):

    train_data_folder = os.path.join(datafolder, 'train')
    test_data_folder = os.path.join(datafolder, 'val')
    train_pert_folder = os.path.join(pertfolder, 'train')
    test_pert_folder = os.path.join(pertfolder, 'val')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = JoinedNet(pretrained_model=pretrained_model, use_unet=use_unet)
    net.to(device)
    # DEBUG: compute sum of parameters of pretrained model
    train_loader = load_data(datafolder=train_data_folder, batch_size=batch_size, aug=use_aug, shuffle=True)
    test_loader = load_data(datafolder=test_data_folder, batch_size=batch_size, aug=False, shuffle=False)
    pert_list = load_pert(train_pert_folder)
    test_pert_list = load_pert(test_pert_folder)
    error, optimizer = build_loss(net, learning_rate=lr)
    cross_entropy_loss = nn.CrossEntropyLoss()
    print("Start training...")
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            # images shape: [batch_size, 3, h, w]
            # labels shape: [batch_size]
            pert = random.choice(pert_list)
            train  = Variable(images).to(device)
            train_pert  = Variable(images + pert).to(device)
            # labels = Variable(labels).to(device)
            y, y_hat, x_filtered = net(train, train_pert)
            # loss = error(y, y_hat)
            # loss = error(y, y_hat) + error(train, x_filtered)
            loss = (
                label_loss_weight * torch.mean((y - y_hat) ** 2)
                +
                recover_loss_weight * torch.mean(abs(train - x_filtered))
            )
            # loss = error(train, x_filtered)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if debug:
                break
        if epoch % print_every == 0:
            print('[{:05}] loss: {:.4f}'.format(epoch, running_loss/(i+1)))
            compare_fool_rate(net, pretrained_model, test_loader, test_pert_list, device)
        if debug:
            break
    if need_save:
        save_all('output/{}_wlab_{}_wrec_{}'.format(dataset, label_loss_weight, recover_loss_weight), net, test_loader, test_pert_list, device, dataset=dataset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default="0")
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--need_save', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--recover_loss_weight', type=float, default=40.0)
    ap.add_argument('--label_loss_weight', type=float, default=1.0)
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # dataset = 'imagenet'
    dataset = 'natural_images'
    datafolder = 'natural_images_subset'
    pertfolder = 'pert_natural_images_subset'
    # dataset = 'cifar10'
    # datafolder = 'CIFAR10'
    # pertfolder = 'pert'

    need_save = args.need_save

    if dataset == 'imagenet':
        SIZE = IMAGENET_SIZE
        STD = IMAGENET_STD
        MEAN = IMAGENET_MEAN
        pretrained_model = torchvision.models.resnet18(pretrained=True)
    elif dataset == 'cifar10':
        SIZE = CIFAR10_SIZE
        STD = CIFAR10_STD
        MEAN = CIFAR10_MEAN
        pretrained_model = models.resnet18(pretrained=True)
    elif dataset == 'natural_images':
        SIZE = IMAGENET_SIZE
        STD = IMAGENET_STD
        MEAN = IMAGENET_MEAN
        import fine_tune
        pretrained_model, _ = fine_tune.load_pretrained()
    
    train_pfn(debug=args.debug, lr=0.005, epochs=args.epochs, n_res_blocks=10, pretrained_model=pretrained_model, use_unet=False, use_aug=True, recover_loss_weight=args.recover_loss_weight, label_loss_weight=args.label_loss_weight, dataset=dataset, datafolder=datafolder, pertfolder=pertfolder, need_save=need_save)