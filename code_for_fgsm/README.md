# Fast Gradient Sign Method

#Usage

 - python3 fgsm_mnist.py --img one.jpg --model model_name

#example:

 - `python3 fgsm_imagenet.py --img orange.jpg --model resnet18`
 or 
 - run with default parameters: `python3 fgsm_imagenet.py`


# any other pretrained model will first be downloaded and will run automatically

#Control keys on front-end
  
  - use trackbar to change `epsilon` (max norm)  
  - `esc` - close  
  - `x` - save perturbation and adversarial image  

