import numpy as np
from PIL import Image
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', action='append', required=True)
    parser.add_argument('-o', '--out_dir', default='output_ensemble')
    parser.add_argument('-v', '--voting_scheme', default='greater') # Options: 'one', 'half', 'greater', 'all'
    args = parser.parse_args()

    n_dirs = len(args.input_directory)
    if n_dirs < 2:
        print('Pass at least 2 input directories!')
        exit()

    for _dir in args.input_directory:
        if not os.path.exists(_dir):
            print('Invalid input directory: ', _dir)
            exit()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.voting_scheme == 'one':
        # Choose pro if at least one votes pro
        majority = 1
    elif args.voting_scheme == 'half':
        # In case of equality choose pro
        majority = (n_dirs + 1) // 2
    elif args.voting_scheme == 'greater':
        # In case of equality choose contra
        majority = (n_dirs + 2) // 2
    elif args.voting_scheme == 'all':
        # Choose pro only if all vote pro
        majority = n_dirs
    else:
        print('Invalid voting scheme!')
        exit()

    img_names = [name for name in os.listdir(args.input_directory[0])]

    for name in img_names:
        mask = np.zeros((608, 608), dtype=np.int)
        for _dir in args.input_directory:
                mask += Image.open(os.path.join(_dir, name))
        mask = np.where(mask >= majority * 255, np.ones_like(mask), np.zeros_like(mask))
        mask = (mask * 255).astype(np.uint8)

        path = os.path.join(args.out_dir, name)
        Image.fromarray(mask).save(path)



if __name__ == "__main__":
    main()