import argparse
import os
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm
from utilities import cross_entropy, compute_accuracy, CustomDataset, get_model


def main():

    # Parse configuration given via command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-te', '--test_path', default='data/test_images')
    parser.add_argument('-tr', '--train_rgb_path', default='data/training/images')
    parser.add_argument('-tgt', '--train_gt_path', default='data/training/groundtruth')
    parser.add_argument('-e', '--n_epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-d', '--decay', type=float, default=0.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-m', '--model', default='resnet101') # Can be resnet50, deeplab50, resnet101, deeplab101
    args = parser.parse_args()

    # Load and prepare data
    train_paths = (args.train_rgb_path, args.train_gt_path)
    training_set = CustomDataset(train_paths=train_paths)
    test_set = CustomDataset(test_path=args.test_path)
    output_path = 'output_' + args.model 
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Get possibly pretrained model from PyTorch
    model = get_model(args.model)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Train and apply model
    train(model, training_set, args)
    test(model, test_set, args)


def train(model, dataset, args):
    data = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

    overall_iter = 0  

    model.train()

    for e in tqdm(range(args.n_epochs)):
        for batch in tqdm(data, total=(100//args.batch_size)):

            tensors = [batch['img'], batch['gt']]
            images, target = tuple([t.cuda() for t in tensors])

            result = model(images)['out']
            mask = result.argmax(1)

            loss = cross_entropy(result, target)
            accuracy = compute_accuracy(target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm.write('Loss: {}'.format(loss.item()))
            tqdm.write('Accuracy: {}\n'.format(accuracy))

            overall_iter += 1


def test(model, dataset, args):
    data = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size)

    print("Applying model on test set...")

    imgize = tv.transforms.ToPILImage()

    with torch.no_grad():
        model.eval()
        for batch in tqdm(data, total=int(np.ceil(94 / args.batch_size))):
            tensors = [batch['img']]
            images, path = tuple([t.cuda() for t in tensors])[0], batch['path']
            outputs = model(images)['out']
            output_mask = outputs.argmax(1)

            for mask, p in zip(output_mask, path):
                _mask = torch.where(mask == 1,
                                    torch.ones_like(mask) * 255,
                                    torch.zeros_like(mask)).byte()
                _mask = imgize(_mask.cpu())
                filename = os.path.basename(p)
                _mask.save(os.path.join(args.output_path, filename))


if __name__ == "__main__":
    main()
