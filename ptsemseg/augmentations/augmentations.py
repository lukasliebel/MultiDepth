# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, lbl_semseg, lbl_side):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            lbl_semseg = Image.fromarray(lbl_semseg, mode="L")
            if not lbl_side.dtype == np.uint32:
                lbl_side = lbl_side.astype(np.uint32)
            lbl_side = Image.fromarray(lbl_side, mode="I")
            self.PIL2Numpy = True

        assert img.size == lbl_semseg.size == lbl_side.size
        for a in self.augmentations:
            img, lbl_semseg, lbl_side = a(img, lbl_semseg, lbl_side)

        if self.PIL2Numpy:
            img, lbl_semseg, lbl_side = np.array(img), np.array(lbl_semseg, dtype=np.uint8), np.array(lbl_side, dtype=np.uint16)

        return img, lbl_semseg, lbl_side


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, lbl_semseg, lbl_side):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            lbl_semseg = ImageOps.expand(lbl_semseg, border=self.padding, fill=0)
            lbl_side = ImageOps.expand(lbl_side, border=self.padding, fill=65535)

        assert img.size == lbl_semseg.size == lbl_side.size
        w, h = img.size
        th, tw = self.size
        # if w == tw and h == th:
        #     print("ERROR: image was just big enough in RandomCrop!")
        #     return img, lbl_semseg, lbl_side
        if w < tw or h < th:
            # return (
                # print("ERROR: image was too small in RandomCrop. Returning upsampled image")
            img = img.thumbnail((tw, th), Image.BILINEAR),
            lbl_semseg = lbl_semseg.thumbnail((tw, th), Image.NEAREST),
            lbl_side = lbl_side.thumbnail((tw, th), Image.BILINEAR),
            # )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print("cropping from", x1, y1, "to", x1 + tw, y1 + th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            lbl_semseg.crop((x1, y1, x1 + tw, y1 + th)),
            lbl_side.crop((x1, y1, x1 + tw, y1 + th)),
        )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), lbl_semseg, lbl_side


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return tf.adjust_saturation(img,
                                    random.uniform(1 - self.saturation,
                                                   1 + self.saturation)), lbl_semseg, lbl_side


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return tf.adjust_hue(img, random.uniform(-self.hue,
                                                  self.hue)), lbl_semseg, lbl_side


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return tf.adjust_brightness(img,
                                    random.uniform(1 - self.bf,
                                                   1 + self.bf)), lbl_semseg, lbl_side

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf)), lbl_semseg, lbl_side

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            lbl_semseg.crop((x1, y1, x1 + tw, y1 + th)),
            lbl_side.crop((x1, y1, x1 + tw, y1 + th))
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, lbl_semseg, lbl_side):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                lbl_semseg.transpose(Image.FLIP_LEFT_RIGHT),
                lbl_side.transpose(Image.FLIP_LEFT_RIGHT)
            )
        return img, lbl_semseg, lbl_side


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, lbl_semseg, lbl_side):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                lbl_semseg.transpose(Image.FLIP_TOP_BOTTOM),
                lbl_side.transpose(Image.FLIP_TOP_BOTTOM)
            )
        return img, lbl_semseg, lbl_side


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        return (
            img.resize(self.size, Image.BILINEAR),
            lbl_semseg.resize(self.size, Image.NEAREST),
            lbl_side.resize(self.size, Image.BILINEAR),
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset  # tuple (delta_x, delta_y)

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(img,
                              y_crop_offset,
                              x_crop_offset,
                              img.size[1]-abs(y_offset),
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
              tf.pad(cropped_img,
                     padding_tuple,
                     padding_mode='reflect'),
              tf.affine(lbl_semseg,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250),
              tf.affine(lbl_side,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=65535))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, lbl_semseg, lbl_side):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            tf.affine(lbl_semseg,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.NEAREST,
                      fillcolor=250,
                      shear=0.0),
            tf.affine(lbl_side,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.BILINEAR,
                      fillcolor=65535,
                      shear=0.0))



class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, lbl_semseg
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                lbl_semseg.resize((ow, oh), Image.NEAREST),
                lbl_side.resize((ow, oh), Image.BILINEAR)
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                lbl_semseg.resize((ow, oh), Image.NEAREST),
                lbl_side.resize((ow, oh), Image.BILINEAR),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                lbl_semseg = lbl_semseg.crop((x1, y1, x1 + w, y1 + h))
                lbl_side = lbl_side.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    lbl_semseg.resize((self.size, self.size), Image.NEAREST),
                    lbl_side.resize((self.size, self.size), Image.BILINEAR),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, lbl_semseg, lbl_side))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, lbl_semseg, lbl_side):
        assert img.size == lbl_semseg.size == lbl_side.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, lbl_semseg, lbl_side = (
            img.resize((w, h), Image.BILINEAR),
            lbl_semseg.resize((w, h), Image.NEAREST),
            lbl_side.resize((w, h), Image.BILINEAR),
        )

        return self.crop(*self.scale(img, lbl_semseg, lbl_side))
