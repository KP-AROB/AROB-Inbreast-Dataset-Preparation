import argparse, os
from glob import glob
from src.preprocessing import *
from src.preparation.image import prepare_inbreast
from src.preparation.classification import bi_rads_classification_preparation
from src.preparation.augmentation import make_augmentation
from src.utils.count import count_images_in_subdirectories, count_png_files
# run: python main.py --data_dir ./data/INbreast --out_dir ./data/INbreast/NormalPNGs --img_size 256
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INbreast benchmark")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--task", type=str, default='bi-rads-cls', choices=['bi-rads-cls', 'lesion-cls', 'segmentation'])
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--synthetize", action='store_true')
    parser.set_defaults(synthetize=False)

    args = parser.parse_args()

    if args.task == 'bi-rads-cls':
        bi_rads_classification_preparation(
            args.data_dir,
            args.out_dir,
            args.img_size,
            args.synthetize
        )
    else:
        prepare_inbreast(
            args.data_dir,
            args.out_dir,
            args.img_size,
            args.task,
            args.synthetize
        )

    if args.task != 'segmentation' and args.augmentation_ratio > 0:
        make_augmentation(args.out_dir + f'/{args.task}', args.augmentation_ratio)
    
    
    print("Number of images in the final dataset : {}".format(count_png_files(args.out_dir)))
    print("Class balance : {}".format(count_images_in_subdirectories(args.out_dir + f'/{args.task}')))
    print("\n\033[32mDone\033[0m\n")
