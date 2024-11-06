import os, cv2
import albumentations as A
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def make_augmentation(data_dir, num_augmentations: int = 100):

    print("\n\033[32mRunning data augmentation ... \033[0m")

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),    
        A.VerticalFlip(p=0.5),    
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.Rotate(limit=90, p=1.0)
    ])

    norm_path = os.path.join(data_dir, 'classification', 'norm')
    abnorm_path = os.path.join(data_dir, 'classification', 'abnorm')

    norm_images = glob(norm_path + '/*.png')
    abnorm_images = glob(abnorm_path + '/*.png')

    def augment_image(image_path):
        image = cv2.imread(image_path)
        augmented_images = []
        for _ in range(num_augmentations):
            augmented = augmentation_pipeline(image=image)['image']
            augmented_images.append(augmented)
        return augmented_images

    with tqdm(total=len(norm_images), desc="Augmenting normal images") as pbar:
        for i, img in enumerate(norm_images):
            augmented_images = augment_image(img)
            for j, augmented_image in enumerate(augmented_images):
                output_path = f"{norm_path}/aug_{i}_{j}.png"
                cv2.imwrite(output_path, augmented_image)
            pbar.update()
    
    with tqdm(total=len(abnorm_images), desc="Augmenting abnormal images") as pbar:
        for i, img in enumerate(abnorm_images):
            augmented_images = augment_image(img)
            for j, augmented_image in enumerate(augmented_images):
                output_path = f"{abnorm_path}/aug_{i}_{j}.png"
                cv2.imwrite(output_path, augmented_image)
            pbar.update()
    
    print("\033[32mAugmentations finished. \033[0m\n")