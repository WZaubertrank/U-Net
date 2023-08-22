import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = Path(mask_dir) / (idx + mask_suffix + '.tif')
    if not mask_file.is_file():
        raise ValueError(f"No mask file found for index {idx}")

    mask = np.asarray(load_image(str(mask_file)))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
# def unique_mask_values(idx, mask_dir, mask_suffix):
    # mask_file = Path(mask_dir) / (idx + mask_suffix + '.tif')
    # if not mask_file.is_file():
    #     raise ValueError(f"No mask file found for index {idx}")

    # mask = np.asarray(load_image(str(mask_file)))
    # if mask.ndim == 2:
    #     unique, counts = np.unique(mask, return_counts=True)
    #     return list(zip(unique, counts))
    # elif mask.ndim == 3:
    #     mask = mask.reshape(-1, mask.shape[-1])
    #     unique, counts = np.unique(mask, axis=0, return_counts=True)
    #     return list(zip(map(tuple, unique), counts))
    # else:
    #     raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

            
        # color_counts = {}
        # for color_count in unique:
        #     for color, count in color_count:
        #         color_counts[color] = color_counts.get(color, 0) + count

        # 选择计数最高的11个颜色
        # self.mask_values = [color for color, _ in sorted(color_counts.items(), key=lambda item: item[1], reverse=True)[:11]]

        # logging.info(f'Unique mask values: {self.mask_values}')

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            label_seg = np.zeros((newH, newW), dtype=np.int64)

            # Define your RGB values and their corresponding labels
            colors_to_labels = {
                tuple([255, 255, 255]): 0,# Non-annotated pixels (should be ignored)
                tuple([255, 0, 255]): 1,# Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included
                tuple([241, 57, 207]): 1,# Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included
                tuple([10, 124, 213]): 2, #Benign mucosa (colonic and ileal)
                tuple([35, 238, 171]): 3,# Tumoral stroma
                tuple([0, 255, 255]): 4,# Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels
                tuple([216, 155, 134]): 4,# Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels
                tuple([255, 102, 102]): 5,# Muscularis propria | Muscularis mucosae
                tuple([176, 157, 127]): 5, # Muscularis propria | Muscularis mucosae
                tuple([10, 254, 6]): 6,# Any forms of lymphatic tissue: lymphatic aggregates, lymph node tissue
                tuple([111, 252, 2]): 7,# Ulceration (surface) | Necrotic debris
                tuple([200, 61, 228]): 8,# Acellular mucin lakes
                tuple([183, 67, 21]): 9,# Bleeding areas – only erythrocytes without any stromal or other tissue
                tuple([71, 50, 10]): 10,# Slide background

             }

            # Map each RGB value to its corresponding label
            for rgb, label in colors_to_labels.items():
                label_seg[(img == rgb).all(-1)] = label

            return label_seg

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

 
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self.mask_dir / (name + self.mask_suffix + '.tif')
        img_file = self.images_dir / (name + '.tif')

        assert img_file.is_file(), f'No image found for the ID {name}: {img_file}'
        assert mask_file.is_file(), f'No mask found for the ID {name}: {mask_file}'

        mask = load_image(str(mask_file))
        img = load_image(str(img_file))

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }



class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='-labelled')