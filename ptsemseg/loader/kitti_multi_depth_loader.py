import os
import imageio
import torch
import pandas as pd
import numpy as np
import scipy.misc as m
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('ptsemseg')


class kittiMultiDepthLoader(data.Dataset):

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(2710, 3384),
        augmentations=None,
        img_norm=True,
        depth_scaling=None,
        max_depth=256,  # absolute dtype d_max for kitti is (2^16 - 1) / 256. empiric dmax for the train set is <116
        min_depth=256,   # empiric dmin for the train set is >2
        n_bins=16
    ):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.depth_scaling = depth_scaling
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.n_bins = n_bins
        self.n_classes_d_cls = self.n_bins
        self.discrete_depth_min = 5  # TODO min depth for discretization in m
        self.discrete_depth_max = 50  # TODO max depth for discretization in m

        # find depth maps in subfolders
        # logger.debug("Searching for KITTI depth files...")
        depth_maps = []

        if self.split in ["train", "val"]:
            parent_dir = os.path.join(self.root, "depth", self.split)
        else:
            parent_dir = None   # FIXME

        for acquisition_dir in os.listdir(parent_dir):
            acquisition_dir = os.path.join(
                                parent_dir,
                                acquisition_dir)
            for camera in ["image_02", "image_03"]:
                camera_dir = os.path.join(acquisition_dir,
                                          "proj_depth",
                                          "groundtruth",
                                          camera)
                for cur_depth_file in os.listdir(camera_dir):
                    cur_depth_file = os.path.join(camera_dir,
                                                  cur_depth_file)
                    depth_maps.append(cur_depth_file)

        # find matching rgb images in raw KITTI data
        # logger.debug("Searching for matching KITTI RGB images...")
        rgb_files = []

        if self.split in ["train", "val"]:
            parent_dir = os.path.join(self.root, "rgb")
        else:
            parent_dir = None   # FIXME

        for cur_depth_map in depth_maps:
            cur_depth_map_ids = cur_depth_map.split("/")
            acquisition_id = cur_depth_map_ids[-5]
            camera_id = cur_depth_map_ids[-2]
            file_id = cur_depth_map_ids[-1]
            rgb_file = os.path.join(parent_dir,
                                    acquisition_id,
                                    camera_id,
                                    "data",
                                    file_id)
            if os.path.isfile(rgb_file):
                rgb_files.append(rgb_file)
            else:
                rgb_files.append(None)

        # remove incomplete pairs
        self.kitti_samples = pd.DataFrame({"rgb": rgb_files,
                                           "depth": depth_maps})
        self.kitti_samples.dropna(inplace=True)
        self.kitti_samples.sort_values(by="rgb", inplace=True)
        self.kitti_samples.reset_index(drop=True, inplace=True)

        # logger.debug("found {} samples for subset {}!".format(len(self.kitti_samples), self.split))

    def __len__(self):
        """
        Get total size of the selected KITTI dataset split.
        """
        return len(self.kitti_samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the KITTI dataset.
        """

        # logger.debug("getting batch number " + str(idx))
        # get paths to rgb and depth files
        img_file_path = self.kitti_samples.loc[idx, 'rgb']
        lbl_d_reg_file_path = self.kitti_samples.loc[idx, 'depth']

        # print("DEBUG: Loading", img_file_path, lbl_d_reg_file_path)

        # read files
        # logger.debug("reading files")
        img = np.array(imageio.imread(img_file_path))
        lbl_d_reg = self.depth_read(lbl_d_reg_file_path)

        # normalize size of images
        # print("image shape:", img.shape)
        # if img.shape != (375, 1244, 3):
        #     img = m.imresize(img, (375, 1244))
        #     print("image shape resized to:", img.shape)
        # print("depth img shape:", lbl_side.shape)
        # if lbl_side.shape != (375, 1244):
        #     lbl_side = m.imresize(lbl_side, (375, 1244), mode="F")
        #     print("depth img shape resized to:", lbl_side.shape)

        # get mask for missing values
        mask = (lbl_d_reg != 0)

        # apply selected data augmentation operations
        # logger.debug("data augmentation")
        if self.augmentations is not None:
            img, _, lbl_d_reg = self.augmentations(img,
                                                  mask,     # HACK
                                                  lbl_d_reg)

        # convert scale of side annotations for training
        # logger.debug("scale depths")
        lbl_d_reg = self.scale_depths(lbl_d_reg)

        # transform image and annotations
        # logger.debug("transform")
        mask = (mask != 0)
        sample = {"image": img, "mask": mask, "d_reg": lbl_d_reg}
        sample = self.transform_multitask(sample)
        mask = sample.pop("mask")
        sample["d_reg"] = sample["d_reg"] * mask  # WARNING depth values == d_min will be treated like missing depths

        # create gt for discrete depth classification
        # logger.debug("discretize depths")
        depth_classes = self.discretize_depths(sample["d_reg"].detach().cpu().numpy())
        sample["d_cls"] = torch.from_numpy(depth_classes).long()

        img = sample.pop("image")
        sample = (img, sample)
        # logger.debug("return sample")
        return sample

    def discretize_depths(self, depth_map):
        bins = np.linspace(0, 1, self.n_bins + 1)
        depth_classes = np.digitize(depth_map, bins, right=True)    # 0 --> 0, 1 --> n_bins
        if depth_classes.max() > self.n_bins:
            logger.error("Class labels > n_classes detected. This is probably due to poorly scaled depth data with d_max_scaled > 1")
        depth_classes[depth_classes == 0] = 251
        depth_classes = depth_classes - 1   # [0, 1, ..., n_bins - 1, 250]
        return depth_classes

    def scale_depths(self, depth_map):

        if self.depth_scaling is not None:
            depth_map_clipped = np.clip(depth_map, self.min_depth, self.max_depth)     # clip to min/max
        depth_map = depth_map_clipped

        if self.depth_scaling == "norm_log":
            depth_map = np.log(depth_map - self.min_depth + 1) / np.log(self.max_depth - self.min_depth + 1)
        else:
            logger.error("Scaling method", self.depth_scaling, "was selected. However, currently only norm_log is available. Depth maps will not be scaled!")

        return depth_map

    def restore_metric_depths(self, depth_map):

        if self.depth_scaling == "norm_log":
            depth_map = np.exp(depth_map * np.log(self.max_depth - self.min_depth + 1)) + self.min_depth - 1
        else:
            logger.error("Scaling method", self.depth_scaling, "was selected. However, currently only norm_log is available. Metric depth maps will not be restored!")

        return depth_map

    def visualize_depths(self, depth_map):
        if len(depth_map.shape) > 2:
            depth_map = depth_map[0, :, :]
        depth_map = depth_map - self.min_depth
        depth_map = depth_map / (self.max_depth - self.min_depth)
        depth_map = np.clip(depth_map, 0, 1)
        cmap = plt.cm.viridis
        depth_map = cmap(depth_map)
        depth_map[depth_map == cmap(0)] = 1
        depth_map = (depth_map * 255).astype(np.uint8)[:, :, 0:3]
        depth_map = np.rollaxis(depth_map, 2, 0)
        return depth_map

    def decode_segmap(self, temp):
        # logger.debug("decoding segmap with shape " + str(temp.shape))
        r = np.ones(temp.shape, dtype=float)
        g = np.ones(temp.shape, dtype=float)
        b = np.ones(temp.shape, dtype=float)
        for l in range(0, self.n_bins + 1):
            color = plt.cm.get_cmap('viridis', self.n_bins)(l)
            r[temp == l] = color[0]
            g[temp == l] = color[1]
            b[temp == l] = color[2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:] = np.nan
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def depth_read(self, filename):
        # Source: KITTI depth devkit
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt

        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = 0.
        return depth

    def transform_multitask(self, sample):

        # Apply transformations to image
        # Resize
        sample["image"] = m.imresize(sample["image"], (self.img_size[0], self.img_size[1]))
        # RGB to BGR
        sample["image"] = sample["image"][:, :, ::-1]
        # To float
        sample["image"] = sample["image"].astype(np.float64)
        # NHWC to NCHW
        sample["image"] = sample["image"].transpose(2, 0, 1)
        # Normalize
        sample["image"] = sample["image"] / 255.0
        # To tensor
        sample["image"] = torch.from_numpy(sample["image"]).float()

        # Apply transformations to mask
        # To float
        sample["mask"] = sample["mask"].astype(float)
        # Resize
        sample["mask"] = m.imresize(sample["mask"], (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        # To int
        sample["mask"] = sample["mask"].astype(int)
        # To tensor
        sample["mask"] = torch.from_numpy(sample["mask"]).float()

        # Apply transformations to mask
        # To float
        sample["d_reg"] = sample["d_reg"].astype(float)
        # Resize
        sample["d_reg"] = m.imresize(sample["d_reg"], (self.img_size[0], self.img_size[1]), mode="F")
        # To tensor
        sample["d_reg"] = torch.from_numpy(sample["d_reg"]).float()

        return sample
