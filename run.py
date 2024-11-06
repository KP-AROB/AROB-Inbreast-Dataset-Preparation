import argparse, os
from glob import glob
from src.preprocessing import *
from src.preparation.image import prepare_inbreast
from src.preparation.augmentation import make_augmentation

# run: python main.py --data_dir ./data/INbreast --out_dir ./data/INbreast/NormalPNGs --img_size 256
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INbreast benchmark")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--task", type=str, default='classification', choices=['classification', 'segmentation'])
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--synthetize", action='store_true')
    parser.set_defaults(synthetize=False)

    args = parser.parse_args()

    prepare_inbreast(
        args.data_dir,
        args.out_dir,
        args.img_size,
        args.task,
        args.synthetize
    )

    if args.task == 'classification' and args.augmentation_ratio > 0:
        make_augmentation(args.out_dir, args.augmentation_ratio)

    norm_path = os.path.join(args.out_dir, 'classification', 'norm')
    abnorm_path = os.path.join(args.out_dir, 'classification', 'abnorm')

    norm_images = glob(norm_path + '/*.png')
    abnorm_images = glob(abnorm_path + '/*.png')

    print('Total dataset length : {}'.format(len(norm_images) + len(abnorm_images)))
    print("\n\033[32mDone\033[0m\n")