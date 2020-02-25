# Universal Perturbation Attack and Defense
This folder contains the source code for the attack (`my_universal.py`) and defense (`model.py`) of universal perturbation.

We don't include the dataset and the trained model since their are too large.


### Attack by computing Universal Perturbation


```
python my_universal.py --delta 0.2 --batch_size 64 -m resnet18 --pretrained_dataset <...> --dataset <...> --gpu 0 --test_accuracy --num_classes 8
```

For example:

```
python my_universal.py --delta 0.2 --batch_size 64 -m resnet18 --pretrained_dataset natural_images --dataset natural_images_subset/train --gpu 0 --test_accuracy --num_classes 8
```

This assume (1) you have a model pretrained on natural_images dataset and (2) the data folder is `natural_images_subset/train`.


### Train Defense Network

Change the following variables in `model.py` according to your need:
```
dataset = 'natural_images'
datafolder = 'natural_images_subset'
pertfolder = 'pert_natural_images_subset'
```

Run `model.py`

```
python model.py --epochs=3  --need_save --gpu 1
```
