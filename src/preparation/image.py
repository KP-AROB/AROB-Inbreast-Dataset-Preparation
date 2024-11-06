import os, shutil
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.preprocessing import *
from concurrent.futures import ProcessPoolExecutor


def process_row(row, data_dir, norm_path, abnormal_path, resize, task, synthetize = False):
    try:
        isNormal = row["Lesion annotation status"] == 0
        folderPath = norm_path if isNormal else abnormal_path
        image_path = glob(
            os.path.join(data_dir, "AllDICOMs", str(row["File name"]) + "*.dcm")
        )[0]
        mask_path = os.path.join(
            data_dir, "AllXML", "{}.xml".format(str(row["File name"]))
        )

        original_image = load_dicom_image(image_path)
        cropped_image, cropped_roi, bounding_box = crop_to_roi(original_image)
        normalized_image = truncate_normalization(cropped_image, cropped_roi)

        if synthetize:
            clahe1_image = clahe(normalized_image, 1.0)
            clahe2_image = clahe(normalized_image, 2.0)
            normalized_image = cv2.merge((normalized_image, clahe1_image, clahe2_image))
        resized_image = cv2.resize(
            normalized_image, (resize, resize), interpolation=cv2.INTER_LINEAR
        )
        # If picture has anomalies and task is set to segmentation : Save the mask
        if not isNormal and task == 'segmentation':
            mask = np.zeros((resize, resize), dtype=np.uint8)
            inbreast_mask = load_inbreast_mask(
                mask_path, imshape=original_image.shape
            )
            if inbreast_mask is not None:
                cropped_mask = crop_img(inbreast_mask, bounding_box)
                mask = cv2.normalize(
                    cropped_mask,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                ).astype(np.uint8)
                mask = cv2.resize(
                    mask, (resize, resize), interpolation=cv2.INTER_LINEAR
                )
                cv2.imwrite(
                    os.path.join(
                        abnormal_path, "masks", "{}.png".format(row["File name"])
                    ),
                    mask,
                )
        save_path = os.path.join(folderPath, "{}.png".format(row["File name"])) if task == 'classification' else  os.path.join(folderPath, "images", "{}.png".format(row["File name"]))
        cv2.imwrite(
            save_path,
            resized_image,
        )
    except Exception as e:
        print(f"Failed to process row {row['File name']}: {e}")

def prepare_inbreast(data_dir, out_dir, resize=256, task='classification', synthetize = False):

    norm_path = os.path.join(out_dir, task, "norm")
    abnormal_path = os.path.join(out_dir, task, "abnorm")
    
    shutil.rmtree(os.path.join(out_dir, task), ignore_errors=True)

    os.makedirs(norm_path, exist_ok=True)
    os.makedirs(abnormal_path, exist_ok=True)

    if task == 'segmentation':
        os.makedirs(os.path.join(norm_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(abnormal_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(abnormal_path, 'masks'), exist_ok=True)

    # Keep patients with mass anomalies
    df = pd.read_excel(os.path.join(data_dir, "INbreast.xls"), skipfooter=2)
    df.columns = df.columns.str.strip().str.capitalize()
    df = df[
        df["Mass"].notna()
        | df[["Mass", "Micros", "Distortion", "Asymmetry"]].isna().all(axis=1)
    ]

    # If cell is NaN then row is abnormal study
    df["Lesion annotation status"] = df["Lesion annotation status"].fillna(1)
    df.loc[df["Lesion annotation status"] != 1, "Lesion annotation status"] = 0

    print("\n\033[32mPreparing INBreast Dataset ... \033[0m\n")
    print("\n\033[32mDataset info\033[0m")
    print("Dataset columns : \n")
    for i in list(df.columns):
        print("    - {}".format(i))
    print("\n")
    print("Dataset length : {}".format(len(df)))

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    process_row,
                    [row for _, row in df.iterrows()],
                    [data_dir] * len(df),
                    [norm_path] * len(df),
                    [abnormal_path] * len(df),
                    [resize] * len(df),
                    [task] * len(df),
                    [synthetize] * len(df)
                ),
                total=len(df),
                desc="Creating images"
            )
        )