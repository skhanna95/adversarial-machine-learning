import os
import importlib
import datetime
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import dataloader
import torchvision
import tqdm
import numpy as np
from PIL import Image
import argparse

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])
CIFAR10_SIZE = 32
classes = None


def idx_to_class(idx: int, labels_path='natural_images.txt'):
    def init_classes(labels_path: str):
        with open(labels_path, 'r') as f:
            return { idx: name for idx, name in enumerate(f.read().split('\n'))}

    global classes
    if classes is None:
        classes = init_classes(labels_path)
    
    return classes[idx]


def attack_single_image(model, img, num_classes=10, max_iter=50, overshoot=0.02, device='cuda'):
    """
    Compute the perturbation for a single image for a given network model.
    model: pretrained network model. (input: [N, 3, 224, 224], output: logits [N, classes]).
    """
    batch = img[None, :].clone().detach().requires_grad_(True)
    logits_np = model(batch)[0]
    top_labels = logits_np.argsort(descending=True)[:num_classes]
    pert_label = gt = top_labels[0]

    pert_img = batch.clone().detach()
    w = torch.zeros(batch.shape).to(device)
    pert_total = torch.zeros(batch.shape).to(device)

    x = batch.clone().detach().requires_grad_(True)
    y = model(x)[0]

    success = False

    for i in range(max_iter):
        if pert_label != gt:
            success = True
            break
        pert = torch.tensor(1e9).to(device)
        y[top_labels[0]].backward(retain_graph=True)
        grad = x.grad.clone().detach()
        for k in range(1, num_classes):
            zero_gradients(x)
            y[top_labels[k]].backward(retain_graph=True)
            grad_k = x.grad.clone().detach()
            w_k = grad_k - grad
            f_k = y[top_labels[k]] - y[top_labels[0]]
            pert_k = f_k.abs() / w_k.norm()
            # print(f"w_k.norm(): {w_k.norm()}")
            
            if pert_k < pert:
                pert = pert_k
                w = w_k
                
        # print(f"[{i}]: w: {w}")
        pert_total = pert_total + (pert + 1e-4) * w / w.norm()

        pert_img = batch + (1 + overshoot) * pert_total
        x = pert_img.clone().detach().requires_grad_(True)
        y = model(x)[0]
        pert_label = y.argmax()


    pert_total = (1 + overshoot) * pert_total
    return pert_total, success, gt, pert_label, pert_img


def predict(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    model:
    img: HWC
    return: index of most possible class
    """
    logits = model(img)
    return logits.argmax(1)


def proj_lp(v: torch.Tensor, xi: torch.Tensor, p: np.float) -> torch.Tensor:

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/v.flatten(1).norm())
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = torch.sign(v) * torch.min(v.abs(), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v


def normalize_img(img):
    # to 0-1
    ma = img.max(axis=(0, 1))
    mi = img.min(axis=(0, 1))
    return (img - mi) / (ma - mi)


def save_pert_to_img(pert, path):
    if isinstance(pert, torch.Tensor):
        pert = pert.cpu().numpy()
    if len(pert.shape) == 4 and pert.shape[0] == 1:
        pert = pert[0]
    if pert.shape[0] == 3:
        pert = np.transpose(pert, (1, 2, 0))
    # Normalize
    pert = (normalize_img(pert) * 255).astype(np.uint8)
    im = Image.fromarray(pert)
    im.save(path)


def test_accuracy(dataloader, model, device):
    n = len(dataloader.dataset)
    n_correct = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        labels_pred = predict(model, imgs)
        n_correct += torch.sum(labels == labels_pred).item()
    return 1.0 * n_correct / n



def save_output(
    dataloader: dataloader.DataLoader, model: torch.nn.Module, img_trans_back,
    device: str, univ_pert=None, output_dir='output'):
    n_imgs = len(dataloader.dataset)
    labels_original = torch.zeros(n_imgs)
    labels_pert = torch.zeros(n_imgs)
    i = 0
    start = 0
    pert_path = os.path.join(output_dir, 'pert.png')
    save_pert_to_img(univ_pert, pert_path)
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        end = start + imgs.shape[0]
        labels_original[start:end] = predict(model, imgs)
        labels_pert[start:end] = predict(model, imgs + univ_pert)
        start = end
        for img in imgs:
            im = img.cpu().numpy()
            im = img_trans_back(img.cpu())  # convert to HWC
            label_name = idx_to_class(int(labels_original[i]))
            im.save(os.path.join(output_dir, '{:03}_orig_{}.jpg'.format(i, label_name)))

            label_name = idx_to_class(int(labels_pert[i]))
            im = img_trans_back((img + univ_pert).cpu())  # convert to HWC
            im.save(os.path.join(output_dir, '{:03}_pert_{}.jpg'.format(i, label_name)))
            i += 1
            

def universal_perturbation(
    dataloader: dataloader.DataLoader, model: torch.nn.Module, device: str,
    delta=0.2, xi=10, max_iter_outer=10, max_iter_inner=50, p=np.inf,
    batch_size=128, need_save=True, num_classes=8):
    """
    model: pretrained model.
    """
    xi = torch.tensor(float(xi)).to(device)

    fooling_rate = 0
    univ_pert = 0

    n_imgs = len(dataloader.dataset)
    n_batches = np.int(np.ceil(1.0 * n_imgs / batch_size))

    for i in range(max_iter_outer):
        if fooling_rate >= 1 - delta:
            # print('type(fooling_rate): ', type(fooling_rate))
            # print('1 - delta): ', type(1 - delta))
            # print('1 - delta: {}'.format(1 - delta))
            # print('fooling_rate >= 1 - delta: {}'.format(fooling_rate >= 1 - delta))
            # print('fooling_rate - (1 - delta): {}'.format(fooling_rate - (1 - delta)))
            print('Finish training with fooling_rate: {}'.format(fooling_rate))
            break
        print('Iteration {:02}'.format(i + 1))
        print('Training...')
        for _, (imgs, _) in enumerate(tqdm.tqdm(dataloader)):
            imgs = imgs.to(device)
            for img in imgs:
                if torch.all(predict(model, img[None, :]) == predict(model, img[None, :] + univ_pert)):
                    pert, success, _, _, _ = attack_single_image(model, img, device=device, num_classes=num_classes)
                    if success:
                        univ_pert = proj_lp(univ_pert + pert.clone().detach(), xi, p)
                        pass
                    else:
                        print('failed on single')

        # Compute the label for original & pert dataset
        labels_original = torch.zeros(n_imgs)
        labels_pert = torch.zeros(n_imgs)

        print('Testing...')
        start = 0
        for _, (imgs, _) in enumerate(tqdm.tqdm(dataloader)):
            imgs = imgs.to(device)
            end = start + imgs.shape[0]
            labels_original[start:end] = predict(model, imgs)
            labels_pert[start:end] = predict(model, imgs + univ_pert)
            start = end
        
        fooling_rate = 1.0 * torch.sum(labels_original != labels_pert).item() / n_imgs
        print("[{:03}]: fooling rate: {:.5f}".format(i, fooling_rate))
    return univ_pert


def load_dataset_from_torchvision(dataset_name, batch_size=4, shuffle=True, limit=0):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((CIFAR10_SIZE, CIFAR10_SIZE)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    data = getattr(torchvision.datasets, dataset_name)(
        'dataset_{}'.format(dataset_name), download=True, train=True, transform=train_transforms)
    if limit > 0:
        indices = list(range(limit))
        data = torch.utils.data.Subset(data, indices)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return data_loader


def load_dataset_from_folder(datafolder, batch_size=4, shuffle=True, limit=0):
    imagenet_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    data = torchvision.datasets.ImageFolder(
        root=datafolder,
        transform=imagenet_transforms
    )
    if limit > 0:
        indices = list(range(limit))
        data = torch.utils.data.Subset(data, indices)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle
    )
    return train_loader


def load_pretrained_model(model_name='resnet18', dataset='imagenet'):
    if dataset.lower() == 'imagenet':
        pretrained_model = getattr(torchvision.models, model_name)(pretrained=True)

    elif dataset.lower() == 'cifar10':
        cifar10_models = importlib.import_module('models')
        pretrained_model = getattr(cifar10_models, model_name)(pretrained=True)
    elif dataset.lower() == 'natural_images':
        import fine_tune
        pretrained_model, _ = fine_tune.load_pretrained()
    else:
        raise ValueError('Not supported dataset: {}'.format(dataset))

    for param in pretrained_model.parameters():
        param.requires_grad = False
    return pretrained_model



def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='test_images', help='Dataset you want we attack.')
    ap.add_argument('--pretrained_dataset', default='ImageNet', help='Dataset that the model is trained on.')
    ap.add_argument('--from_torchvision', action='store_true', help='Do we want to load dataset from torchvision')
    ap.add_argument('-m', '--model', default='resnet18', help='Model name. Default: resnet18')
    ap.add_argument('--limit', type=int, default='0', help='Number of data you want to train on')
    ap.add_argument('--save_limit', type=int, default=None, help='Number of data you want to save (to visualize)')
    ap.add_argument('--num_classes', type=int, default=8, help='Number of data you want to save (to visualize)')
    ap.add_argument('-p', '--pert_file', default=None, help='Pert file if you want to skip training.')
    ap.add_argument('--delta', type=float, default=0.1, help='Non-Fooling rate')
    ap.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    ap.add_argument('--gpu', default="0", help='Which GPU')
    ap.add_argument('--need_save', action='store_true', help='Save the images w/ pert and labels')
    ap.add_argument('--test_accuracy', action='store_true', help='Save the images w/ pert and labels')
    ap.add_argument('-o', '--output_dir', default='output', help='Output folder for saved results')
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model_name = args.model
    if args.from_torchvision:
        dataloader = load_dataset_from_torchvision(args.dataset, batch_size=args.batch_size, limit=args.limit)
    else:
        dataloader = load_dataset_from_folder(args.dataset, batch_size=args.batch_size, limit=args.limit)
    model = load_pretrained_model(model_name, args.pretrained_dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if args.test_accuracy:
        print('The accuracy of the model on the dataset: {}'.format(test_accuracy(dataloader, model, device)))


    if args.pert_file:
        pert = np.load(args.pert_file).astype(np.float32)
    else:
        pert = universal_perturbation(dataloader, model, device, delta=args.delta, num_classes=args.num_classes)
        np.save('universal_pert_{}_{}_limit_{}_delta_{}_{}.npy'.format(args.dataset.replace('/', '_'), model_name, args.limit, args.delta, datetime.datetime.strftime(datetime.datetime.now(), "%H_%M_%S")), pert.cpu().detach().numpy())

    if len(pert.shape) == 4 and pert.shape[0] == 1:
        pert = pert[0]
    if pert.shape[2] == 3:
        pert = np.transpose(pert, (2, 0, 1))
    pert = pert.clone().detach().to(device) if isinstance(pert, torch.Tensor) else torch.tensor(pert).to(device)

    clip = lambda x: clip_tensor(x, 0, 1.0)

    img_trans_back = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0, 0, 0], std=1.0/IMAGENET_STD),
        torchvision.transforms.Normalize(mean=-IMAGENET_MEAN, std=[1, 1, 1]),
        torchvision.transforms.Lambda(clip),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(224)])
    if args.need_save:
        save_limit = args.save_limit or args.limit
        output_dir = os.path.join(args.output_dir, args.dataset.replace('/', '_'), args.model)
        print('Saving Results to {}...'.format(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        if args.from_torchvision:
            img_trans_back = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=[0, 0, 0], std=1.0/CIFAR10_STD),
                torchvision.transforms.Normalize(mean=-CIFAR10_MEAN, std=[1, 1, 1]),
                torchvision.transforms.Lambda(clip),
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(CIFAR10_SIZE)])
            dataloader = load_dataset_from_torchvision(args.dataset, batch_size=args.batch_size, limit=save_limit, shuffle=False)
        else:
            img_trans_back = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=[0, 0, 0], std=1.0/IMAGENET_STD),
                torchvision.transforms.Normalize(mean=-IMAGENET_MEAN, std=[1, 1, 1]),
                torchvision.transforms.Lambda(clip),
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(224)])
            dataloader = load_dataset_from_folder(args.dataset, batch_size=args.batch_size, shuffle=False, limit=save_limit)
        save_output(dataloader, model, img_trans_back, device=device, univ_pert=pert, output_dir=output_dir)
    print('Done.')


if __name__ == "__main__":
    main()