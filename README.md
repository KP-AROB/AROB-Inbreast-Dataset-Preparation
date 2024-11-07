# INBreast dataset preparation

This repo aims to prepare the INBreast dataset for an unsupervised breast anomaly dectection task.
In this preparation, the "Mass" class is the only one kept in the dataset to define abnormal images.

Each image can be normalized and synthetized using the CLAHE algorithm. By default Anisotropic Diffusion is applied to every scan to smooth them out and remove noise.
For lower data usage, a flag is also provided to resize the images (default to 256 if not set).

## 1. Requirements

An env.yaml file is provided to install a conda environment containing the setup configuration for this project. To create the environment, run :

```bash
conda env create -f env.yaml
```

The pip dependencies for this project can also be downloaded again using :

```bash
pip install -r requirements.txt
```

## 2. Installation

First, you need to download the dataset. It is available on Kaggle (it can be downloaded from other sources as well) : [Kaggle INBreast dataset](https://www.kaggle.com/datasets/tommyngx/inbreast2012)

## 3. Usage

You can run the preparation script with the following command and given flags : 

```bash
python run.py --data_dir ./data/INbreast --out_dir ./data/INbreast/PNGs
```

| Flag                  | Description                                                                                       | Default Value   |
|-----------------------|---------------------------------------------------------------------------------------------------|-----------------|
| --data_dir            | The folder where the INBreast dataset is stored                                                   | None            |
| --out_dir             | The folder where the prepared dataset will be stored                                              | None            |
| --img_size            | The size to which the image should be resized                                                     | 256             |
| --task                | The task for which the dataset will be prepared ('bi-rads-cls', 'lesion-rads-cls', 'segmentation) | 'bi-ras-cls'    |
| --augmentation_ratio  | The number of augmented image per scan to create                                                  | 10              |
| --synthetize          | Whether to use CLAHE normalization to obtain synthetized images                                   | False           |

The ```--synthetize``` option can be set to true by simple adding the flag to the command as follows : 

```bash
python run.py --data_dir ./data/INbreast --out_dir ./data --synthetize
```

### 3.1. Data augmentation

As the INbreast dataset is relatively small, it can be augmented during the preparation following a pre-defined pipeline, the augmentation can be called with the ```--augmentation_ratio``` flag.
This ratio controls the amount of new images (per scan) that will created. By default this flag has a value of 0 meaning that the dataset will not be augmented.

This augmentation process uses the albumentations library and the augmentation pipeline follows the following code : 

```python

transform = A.Compose([
    A.HorizontalFlip(p=0.5),    
    A.VerticalFlip(p=0.5),    
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.Rotate(limit=90, p=1.0)
])

```

For now, augmentations are only available for the classification task.

### 3.2. File structure

Note that the dataset can be prepared for three different tasks : ```bi-rads-cls```,  ```lesion-cls``` and ```segmentation```.
This choice is given by the ```--task``` flag. The masks are extracted from the dataset only when ```segmentation``` is chosen.
The data folder structure will then be : 

- 📂 data/
    - 📂 segmentation/
        - 📂 abnorm/
            - 📂 images/
                - 📄 01.png
                - 📄 02.png
            - 📂 masks/
                - 📄 01.png
                - 📄 02.png
        - 📂 norm/
            - 📂 images/
                - 📄 01.png
                - 📄 02.png

For the two other tasks the structure will be :

- 📂 data/
    - 📂 bi-rads-cls/
        - 📂 1/
            - 📄 01.png
        - 📂 2/
            - 📄 01.png
        - 📂 3/
            - 📄 01.png
        - 📂 4a/
            - 📄 01.png
        - 📂 4b/
            - 📄 01.png
        - 📂 4c/
            - 📄 01.png
        - 📂 5/
            - 📄 01.png
        - 📂 6/
            - 📄 01.png

