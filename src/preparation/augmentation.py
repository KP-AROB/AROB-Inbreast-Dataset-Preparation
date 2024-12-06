import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm


def make_augmentation(data_dir, num_augmentations: int = 100):

    print("\n\033[32mRunning data augmentation ... \033[0m")

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.5),
        A.GridDistortion(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ])

    label_folders = glob(os.path.join(data_dir, '*'))

    def augment_image(image_path):
        image = cv2.imread(image_path)
        augmented_images = []
        for _ in range(num_augmentations):
            augmented = augmentation_pipeline(image=image)['image']
            augmented_images.append(augmented)
        return augmented_images

    for label in label_folders:
        number_of_images = glob(label + '/*.png')
        with tqdm(total=len(number_of_images), desc=f"Augmenting {label} images") as pbar:
            for i, img in enumerate(number_of_images):
                augmented_images = augment_image(img)
                for j, augmented_image in enumerate(augmented_images):
                    output_path = f"{label}/aug_{i}_{j}.png"
                    cv2.imwrite(output_path, augmented_image)
                pbar.update()

    print("\033[32mAugmentations finished. \033[0m\n")
