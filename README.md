# cil2020-road-segmentation

Team Member: Xiao'ao Song, Weiyi Wang, Pavel Pozdnyakov, Dominik Alberto

This repository contains the tools and models for the the course project of 
[Computational Intelligence Lab](http://da.inf.ethz.ch/teaching/2020/CIL/) (Spring 2020): Road Segmentaion.

Credit: [TorchSeg](https://github.com/ycszen/TorchSeg/) for the structure of repository

## Prerequisites
- PyTorch >= 1.0
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`
- tqdm
  - `pip3 install tqdm`

## Model directory:

```shell
├── config.py
├── dataloader.py
├── eval.py
├── network.py
├── pred.py
└── train.py
```

### Prepare data
Before training the model, you will need to place the training data under ../CIL_Road_Seg/data/training/ and place the test data under ../CIL_Road_Seg/data/training/test_images
 
### A sample workflow:

```shell
cd model/cil-resnet50
# train with CUDA device 0
python train50.py -d 0
# eval using the default last epoh
python eval.py -d 0 -p ./val
# generate predicted groundtruth
python pred.py -d 0 -p ./pred
# generate submission.csv
python ../../data/mask_to_submission.py --name submission -p ./pred/
# submit the submission.csv generated
```

