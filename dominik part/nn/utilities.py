import torch
import os
import numpy as np
import torchvision as tv
from PIL import Image

def get_model(str_model):
    if str_model == 'resnet50':
        model = tv.models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1))
    elif str_model == 'resnet101':
        model = tv.models.segmentation.fcn_resnet101(pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1))
    elif str_model == 'deeplab50':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
    elif str_model == 'deeplab101':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))

    for param in model.parameters():
            param.requires_grad = True

    return model.cuda()

def cross_entropy(p, q):

    mod_p = p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
    mod_q = torch.squeeze(q).view(-1)

    return torch.nn.functional.cross_entropy(mod_p,
                            mod_q,
                            reduction='sum',
                            ignore_index=250)

def compute_accuracy(prediction, groundtruth):

    _gt = groundtruth.cpu().numpy()
    _pred = prediction.cpu().numpy()

    accumulator = np.zeros(4)

    for gt, pr in zip(_gt, _pred):
        flat_gt = gt.flatten()
        flat_pr = pr.flatten()
        temp = np.bincount(2 * flat_gt.astype(int) + flat_pr, minlength=4)
        accumulator += temp

    accuracy = (accumulator[0] + accumulator[-1]) / accumulator.sum()
    return accuracy

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, train_paths=None, test_path=None) -> None:

        if train_paths:
            self.is_training_set = True
            img_dir, gt_dir = train_paths
            self.gt_paths = np.asarray([
                os.path.join(gt_dir, filename)
                for filename in os.listdir(gt_dir)
            ])
            self.gt_paths.sort()
        else:
            self.is_training_set = False
            img_dir = test_path

        self.img_paths = np.asarray([
                os.path.join(img_dir, filename)
                for filename in os.listdir(img_dir)
            ])
        self.img_paths.sort()

        self.length = len(self.img_paths)

        
    def __len__(self):
        ''' Needs to be implemented for Torch's DataLoader '''

        return self.length

    def __getitem__(self, index):
        ''' Needs to be implemented for Torch's DataLoader '''

        item = dict()

        path = self.img_paths[index]
        img = Image.open(path)

        if not self.is_training_set:
            preprocesser = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            item.update([('img', preprocesser(img)), ('path', path)])

            return item

        gt = Image.open(self.gt_paths[index])

        img, gt = self.preprocess(img, gt)

        gt = tv.transforms.functional.to_tensor(gt)
        gt = torch.where(gt < 0.8, torch.zeros_like(gt),
                           torch.ones_like(gt))
        gt = gt.to(torch.int64)

        item.update([('img', img), ('gt', gt), ('path', path)])

        return item

    def preprocess(self, img, gt):

        # First select part, directiction, etc. of image and gt
        r = np.random.randint(2, size=4)
        
        if r[0]:
            deg = (np.random.rand(1) - 0.5) * 30
            img = tv.transforms.functional.rotate(img, deg)
            gt = tv.transforms.functional.rotate(gt, deg)

        if r[1]:
            img = tv.transforms.functional.vflip(img)
            gt = tv.transforms.functional.vflip(gt)

        if r[2]:
            img = tv.transforms.functional.hflip(img)
            gt = tv.transforms.functional.hflip(gt)

        if r[3]:
            img = tv.transforms.Resize(size=(450, 450))(img)
            gt = tv.transforms.Resize(size=(450, 450))(gt)
            x, y, height, width = tv.transforms.RandomCrop.get_params(img,
                                                      output_size=(400, 400))
            img = tv.transforms.functional.crop(img, x, y, height, width)
            gt = tv.transforms.functional.crop(gt, x, y, height, width)

        snd_preprocessor = tv.transforms.Compose([
            tv.transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            tv.transforms.RandomGrayscale(0.1),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = snd_preprocessor(img)
        

        return img, gt