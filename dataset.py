import imgaug.augmenters as iaa
import json
import numpy as np
import os
from PIL import Image, ImageFilter
import random


def load_image(file):
    return Image.open(file).convert("RGB")


class ImageDataset:
    def __init__(
        self,
        image_dir,
        img_to_uid_map,
        polar=False,
        input_transform=None,
        augment=True,
        flip=False,
        skip_not_found=False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.polar = polar
        self.skip_not_found = skip_not_found
        self.flip = flip

        self.img_to_uid = {}
        with open(img_to_uid_map, "r") as fp:
            self.img_to_uid_old = json.load(fp)
            for image_name in self.img_to_uid_old.keys():
                image_name_new = image_name.split(".")[0] + "_p.png"
                self.img_to_uid[image_name_new] = self.img_to_uid_old[image_name]

        uids = set()

        for image_name in self.img_to_uid.keys():
            uid = self.img_to_uid[image_name]
            uids.add(uid)
            if self.flip:
                uids.add(uid + "_flip")

        uids = list(uids)
        self.uid_to_label = {}
        for i in range(len(uids)):
            self.uid_to_label[uids[i]] = i

        self.input_transform = input_transform
        self.augment = augment

        self.image_paths = []

        for image_name in self.img_to_uid.keys():
            image_path = os.path.join(image_dir, image_name)
            self.image_paths.append(image_path)

        self.length = len(self.image_paths)
        self.heavy_augment_prob = 0.3

    def set_augment(self, value):
        print("Augmenting?", value)
        self.augment = value

    def set_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print("Heavy Augmentation Probability set to:", self.heavy_augment_prob)

    def __getitem__(self, index):
        # Get the image and the mask

        if self.skip_not_found:
            while not os.path.exists(self.image_paths[index]):
                index += 1

        image_path = self.image_paths[index]
        image = load_image(image_path)
        uid = self.img_to_uid[os.sep.join(image_path.split(os.sep)[-2:])]
        label = self.uid_to_label[uid]

        # Data augmentation
        if self.augment:
            # horizontal flip
            if random.random() < 0.5 and self.flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = self.uid_to_label[uid + "_flip"]
            # Affine Transformations
            if not self.polar:
                aug = iaa.Affine(
                    scale=(0.8, 1.25),
                    translate_px={"x": (-15, 15), "y": (-15, 15)},
                    rotate=(-30, 30),
                    mode="constant",
                    cval=random.randint(0, 255),
                )
            else:
                w, h = image.size
                aug = iaa.Affine(
                    translate_px={"x": (-int(w / 2), int(w / 2))}, mode="wrap"
                )
            # print(np.expand_dims(np.array(image), axis=0).shape)
            img_np = aug(images=np.expand_dims(np.array(image), axis=0))
            image = Image.fromarray(img_np[0])

            if random.random() < self.heavy_augment_prob:
                random_choice = np.random.choice([1, 2, 3, 4, 5, 6])
                if random_choice == 1:
                    # sharpening
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
                    # blurring
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
                # Basic compression and expansion
                elif random_choice == 4:
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
                        # JPEG compression
                        aug = iaa.JpegCompression(compression=(5, 50))
                        img_np = aug(images=np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                else:  # random contrast change
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
