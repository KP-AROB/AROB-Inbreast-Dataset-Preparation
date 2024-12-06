import os
import shutil
import cv2
import pandas as pd
from src.preparation.log import log_dataset_statistics
from src.preprocessing.load import load_dicom_image
from src.preprocessing.normalize import truncate_normalization
from src.preprocessing.enhance import clahe, anisotropic_diffusion
from src.preprocessing.crop import crop_to_roi
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_bi_rads_row(row, data_dir, out_dir, resize, synthetize=False):
    try:
        label_folder = str(row["Bi-rads"])
        image_save_path = os.path.join(out_dir, 'bi-rads-cls', label_folder)
        image_path = glob(
            os.path.join(data_dir, "AllDICOMs", str(
                row["File name"]) + "*.dcm")
        )[0]

        original_image = load_dicom_image(image_path)
        cropped_image, cropped_roi, _ = crop_to_roi(original_image)
        normalized_image = truncate_normalization(cropped_image, cropped_roi)

        if synthetize:
            clahe1_image = clahe(normalized_image, 1.0)
            clahe2_image = clahe(normalized_image, 2.0)
            normalized_image = cv2.merge(
                (normalized_image, clahe1_image, clahe2_image))

        resized_image = cv2.resize(
            normalized_image, (resize, resize), interpolation=cv2.INTER_LINEAR
        )

        resized_image = anisotropic_diffusion(resized_image)
        save_path = os.path.join(
            image_save_path, "{}.png".format(row["File name"]))

        cv2.imwrite(
            save_path,
            resized_image,
        )
    except Exception as e:
        print(f"Failed to process row {row['File name']}: {e}")


def bi_rads_classification_preparation(data_dir, out_dir, resize=256, synthetize=False):
    task = "bi-rads-cls"
    shutil.rmtree(os.path.join(out_dir, task), ignore_errors=True)
    df = pd.read_excel(os.path.join(data_dir, "INbreast.xls"), skipfooter=2)
    df.columns = df.columns.str.strip().str.capitalize()
    df = df[df["Bi-rads"].notna()]
    print('Classes : {}'.format(df["Bi-rads"].unique()))
    for i in list(df["Bi-rads"].unique()):
        os.makedirs(os.path.join(
            out_dir, 'bi-rads-cls', str(i)), exist_ok=True)
    log_dataset_statistics(df)

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    process_bi_rads_row,
                    [row for _, row in df.iterrows()],
                    [data_dir] * len(df),
                    [out_dir] * len(df),
                    [resize] * len(df),
                    [synthetize] * len(df)
                ),
                total=len(df),
                desc="Creating Bi-rads segmented dataset"
            )
        )
