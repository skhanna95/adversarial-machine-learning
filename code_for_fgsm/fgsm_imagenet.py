import numpy as np
import cv2 as frt_end
import argparse
from imagenet_labels import classes
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms



def main():
    img_SIZE = 224
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]

    print('Fast Gradient Sign Method')
    print('Model: %s' %(model_name))
    print()


    def nothing(z):
        pass

    window_advers = 'perturbation'
    frt_end.namedWindow(window_advers)
    frt_end.createTrackbar('eps', window_advers, 1, 255, nothing)

    def calc_img(image_dir):
        # load image and reshape to (3, 224, 224) and RGB (not BGR)
        # preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
        orig_img = frt_end.imread(image_dir)[..., ::-1]
        orig_img = frt_end.resize(orig_img, (img_SIZE, img_SIZE))
        image = orig_img.copy().astype(np.float32)
        perturb = np.empty_like(orig_img)
        image = image/255.0
        image = (image - mean)/sd
        image = image.transpose(2, 0, 1)
        return image,perturb

    img,perturbation=calc_img(image_dir)

    # GET model from input
    model = getattr(models, model_name)(pretrained=True)
    model.eval()
    cel = nn.CrossEntropyLoss()

    device = 'cpu'


    # prediction for original model
    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

    out_est = model(inp)

    prediction_orig = np.argmax(out_est.data.cpu().numpy())
    print('ORIGINAL CLASS: %s' %(classes[prediction_orig].split(',')[0]))
    _, index = torch.max(out_est, 1)
    percentage = torch.nn.functional.softmax(out_est, dim=1)[0] * 100
    print(percentage[index[0]].item())

    def deprrocess_img(inpt_img):
        # deprocess image
        adv_input = inp.data.cpu().numpy()[0]
        pert = (adv_input - img).transpose(1, 2, 0)
        adv_input = adv_input.transpose(1, 2, 0)
        adv_input = (adv_input * sd) + mean
        adv_input = adv_input * 255.0
        adv_input = adv_input[..., ::-1] 
        adv_input = np.clip(adv_input, 0, 255).astype(np.uint8)
        pert = pert * 255
        pert = np.clip(pert, 0, 255).astype(np.uint8)
        return adv_input,pert

    while True:
        # get trackbar position
        eps = frt_end.getTrackbarPos('eps', window_advers)

        inp = Variable(torch.from_numpy(img).float().unsqueeze(0), requires_grad=True)


        out_est = model(inp)
        loss = cel(out_est, Variable(torch.Tensor([float(prediction_orig)]).to(device).long()))


        # compute gradients
        loss.backward()



        # fgsm-finding sign
        inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))
        inp.grad.data.zero_() 



        # predict adversarial
        pred_adv = np.argmax(model(inp).data.cpu().numpy())

        print(" "*65, end='\r')
        print("After attack: eps [%f] \t%s" %(eps, classes[pred_adv].split(',')[0]), end="\r")


        
        adv,perturbation=deprrocess_img(inp)
        

        # frontend
        frt_end.imshow(window_advers, perturbation)
        frt_end.imshow('adversarial image', adv)
        key = frt_end.waitKey(510) & 0xFF
        if key == 27:
            break
        elif key == ord('x'):
            frt_end.imwrite('adversarial_image.png', adv)
            frt_end.imwrite('added_perturbation.png', perturbation)
    print()
    frt_end.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='orange.jpg', help='path to image')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50','resnet101','vgg16','alexnet'], required=False, help="Which network?")


    args = parser.parse_args()
    image_dir = args.img
    model_name = args.model

    main()

