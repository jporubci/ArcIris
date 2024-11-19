from PIL import Image, ImageFilter
from tqdm import tqdm
import cv2
import imgaug.augmenters as iaa
import json
import numpy as np
import os
import random


class ImageDataset:
    def __init__(self, image_dir, img_to_uid_map, polar=False, input_transform=None, augment=True, flip=False, skip_not_found=False):
        super().__init__()
        self.image_dir = image_dir
        self.polar = polar
        self.skip_not_found = skip_not_found
        self.flip = flip

        with open(img_to_uid_map, "r") as file:
            self.img_to_uid_old = json.load(file)

        s = {}
        for image_path, uid in tqdm(self.img_to_uid_old.items()):
            directory = os.path.join(image_dir, os.path.dirname(image_path))
            filename, ext = os.path.splitext(os.path.basename(image_path))
            if polar:
                filename += "_p"
            full_image_path = os.path.join(directory, f"{filename}{ext}")
            full_image_mask_path = os.path.join(directory, f"{filename}{'m' if polar else '_m'}{ext}")
            if os.path.exists(full_image_path) and os.path.exists(full_image_mask_path):
                s[image_path] = uid
        self.img_to_uid_old = s

        self.img_to_uid = {f"{image_name.split('.')[0]}{'_p' if polar else ''}.png": uid for image_name, uid in self.img_to_uid_old.items()}

        uids = {uid for uid in self.img_to_uid.values()} | {f"{uid}_flip" for uid in self.img_to_uid.values() if self.flip}
        self.uid_to_label = {uid: idx for idx, uid in enumerate(uids)}

        self.input_transform = input_transform
        self.augment = augment
        self.image_paths = [os.path.join(image_dir, image_name) for image_name in self.img_to_uid]
        self.length = len(self.image_paths)
        self.heavy_augment_prob = 0.3


    def set_augment(self, value):
        self.augment = value
        print(f"self.augment: {self.augment}")


    def set_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print(f"self.heavy_augment_prob: {self.heavy_augment_prob}")


    def __getitem__(self, index):
        # Get the image and the mask
        if self.skip_not_found:
            while not os.path.exists(self.image_paths[index]):
                print(self.image_paths[index])
                index += 1

        image_path = self.image_paths[index]
        uid = self.img_to_uid[os.sep.join(image_path.split(os.sep)[-2:])]
        label = self.uid_to_label[uid]

        image = Image.open(image_path).convert("RGB")

        directory, filename = os.path.split(image_path)
        filename, ext = os.path.splitext(filename)
        mask_filename = f"{filename}{'m' if self.polar else '_m'}{ext}"
        mask_path = os.path.join(directory, mask_filename)
        image_mask = Image.open(mask_path).convert("1")
        image_mask = np.array(image_mask)

        blurred_image = np.array(image)
        blurred_image[image_mask == 0] = cv2.GaussianBlur(blurred_image[image_mask == 0], (31, 31), 5)
        image = Image.fromarray(blurred_image)

        # Data augmentation
        if self.augment:
            if random.random() < 0.5 and self.flip:
                # Horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = self.uid_to_label[f"{uid}_flip"]
            # Affine transformations
            if not self.polar:
                aug = iaa.Affine(scale=(0.8, 1.25), translate_px={"x": (-15, 15), "y": (-15, 15)}, rotate=(-30, 30), mode="constant", cval=random.randint(0, 255))
            else:
                w = image.size[0] // 2
                aug = iaa.Affine(translate_px={"x": (-w, w)}, mode="wrap")
            img_np = aug(images=np.expand_dims(np.array(image), axis=0))
            image = Image.fromarray(img_np[0])

            if random.random() < self.heavy_augment_prob:
                random_choice = np.random.choice([1, 2, 3, 4, 5, 6])
                if random_choice == 1:
                    # Sharpening
                    random_degree = np.random.choice([1, 2, 3, 4, 5])
                    if random_degree == 1:
                        image = image.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        image = image.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        image = image.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                elif random_choice == 2:
                    # Blurring
                    random_degree = np.random.choice([1, 2, 3])
                    if random_degree == 1:
                        aug = iaa.AverageBlur(k=(2, 3))
                    elif random_degree == 2:
                        aug = iaa.GaussianBlur(sigma=(0.0, 1.0))
                    else:
                        aug = iaa.MotionBlur(k=3)
                    img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                    image = Image.fromarray(img_np[0])
                elif random_choice == 3:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=5)
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=5)
                    img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                    image = Image.fromarray(img_np[0])
                elif random_choice == 4:
                    # Basic compression and expansion
                    if random.random() < 0.5:
                        divider = random.random() + 1.2
                        cw, ch = image.size
                        new_cw = int(cw / divider)
                        new_ch = int(ch / divider)

                        first_choice = np.random.choice([1, 2, 3, 4, 5, 6])
                        if first_choice == 1:
                            image = image.resize((new_cw, new_ch), Image.NEAREST)
                        elif first_choice == 2:
                            image = image.resize((new_cw, new_ch), Image.BILINEAR)
                        elif first_choice == 3:
                            image = image.resize((new_cw, new_ch), Image.BICUBIC)
                        elif first_choice == 4:
                            image = image.resize((new_cw, new_ch), Image.LANCZOS)
                        elif first_choice == 5:
                            image = image.resize((new_cw, new_ch), Image.HAMMING)
                        else:
                            image = image.resize((new_cw, new_ch), Image.BOX)

                        second_choice = np.random.choice([1, 2, 3, 4, 5, 6])
                        if second_choice == 1:
                            image = image.resize((cw, ch), Image.NEAREST)
                        elif second_choice == 2:
                            image = image.resize((cw, ch), Image.BILINEAR)
                        elif second_choice == 3:
                            image = image.resize((cw, ch), Image.BICUBIC)
                        elif second_choice == 4:
                            image = image.resize((cw, ch), Image.LANCZOS)
                        elif second_choice == 5:
                            image = image.resize((cw, ch), Image.HAMMING)
                        else:
                            image = image.resize((cw, ch), Image.BOX)
                    else:
                        aug = iaa.JpegCompression(compression=(5, 50))
                        img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                else:
                    # Random contrast change
                    random_degree = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
                    if random_degree == 1:
                        aug = iaa.GammaContrast((0.5, 1.0))
                    elif random_degree == 2:
                        aug = iaa.LinearContrast((0.8, 1.2))
                    elif random_degree == 3:
                        aug = iaa.SigmoidContrast(gain=(3, 5), cutoff=(0.4, 0.6))
                    elif random_degree == 4:
                        aug = iaa.LogContrast(gain=(0.8, 1.2))
                    elif random_degree == 5:
                        aug = iaa.pillike.Autocontrast()
                    else:
                        aug = iaa.pillike.EnhanceBrightness()
                    img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                    image = Image.fromarray(img_np[0])

        image = image.convert("L")
        if self.input_transform is not None:
            image = self.input_transform(image)

        data = {"images": image, "labels": label}
        return data


    def __len__(self):
        return self.length
