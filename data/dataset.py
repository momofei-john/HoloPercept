import os
import cv2
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from data.omni_dataset import load_npy_and_npz

# prompt info dict
# position prompt
from data.omni_dataset import position_prompt_dict

# type prompt
from data.omni_dataset import type_prompt_dict

# organ prompt
from data.omni_dataset import position_prompt_one_hot_dict

# type prompt
from data.omni_dataset import type_prompt_one_hot_dict

# mode prompt
from data.omni_dataset import mode_prompt_one_hot_dict


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_horizontal_flip(image, label=None):
    axis = 1
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    else:
        return image


def random_crop_resize(image, label=None):
    ori_x, ori_y, _ = image.shape
    scale = random.uniform(0.8, 1.2)
    image = zoom(image, (scale, scale, 1), order=1)
    if label is not None:
        label = zoom(label, (scale, scale), order=0)

    if scale < 1:
        x, y, _ = image.shape
        startx = ori_x // 2 - (x // 2)
        starty = ori_y // 2 - (y // 2)
        new_image = np.zeros((ori_x, ori_y, 3))
        new_image[startx: startx + x, starty: starty + y, :] = image
        image = new_image
        if label is not None:
            new_label = np.zeros((ori_x, ori_y))
            new_label[startx: startx + x, starty: starty + y] = label
            label = new_label
    else:
        x, y, _ = image.shape
        startx = x // 2 - (ori_x // 2)
        starty = y // 2 - (ori_y // 2)
        image = image[startx: startx + ori_x, starty: starty + ori_y, :]
        if label is not None:
            label = label[startx: startx + ori_x, starty: starty + ori_y]

    if label is not None:
        return image, label
    else:
        return image


class RandomGenerator(object):
    def __init__(self, output_size, no_mask=False):
        self.output_size = np.array(output_size)
        self.no_mask = no_mask

    def __call__(self, sample):

        image = sample["image"]
        is_video = (image.ndim == 4)

        if is_video:
            T, H, W, C = image.shape
            label = None if self.no_mask else sample["label"]
            scale = self.output_size / min(H, W)
            image = zoom(image, (1, scale[0], scale[1], 1), order=1)
            if not self.no_mask:
                label = zoom(label, (1, scale[0], scale[1]), order=0)
        else:
            H, W, C = image.shape
            T = 1
            image = image  # (1,H,W,C)
            label = None if self.no_mask else sample["label"]
            scale = self.output_size / min(H, W)
            image = zoom(image, (scale[0], scale[1], 1), order=1)
            if not self.no_mask:
                label = zoom(label, (scale[0], scale[1]), order=0)

        if "mode_prompt" in sample:
            mode_prompt = sample["mode_prompt"]

        if is_video:
            _, x, y, _ = image.shape
            startx = x // 2 - (self.output_size[0] // 2)
            starty = y // 2 - (self.output_size[1] // 2)
            image = image[
                    :,
                    startx: startx + self.output_size[0],
                    starty: starty + self.output_size[1],
                    :
                    ]
            if not self.no_mask:
                label = label[
                        :,
                        startx: startx + self.output_size[0],
                        starty: starty + self.output_size[1],
                        ]
        else:
            x, y, _ = image.shape
            startx = x // 2 - (self.output_size[0] // 2)
            starty = y // 2 - (self.output_size[1] // 2)
            image = image[
                    startx: startx + self.output_size[0],
                    starty: starty + self.output_size[1],
                    :
                    ]
            if not self.no_mask:
                label = label[
                        startx: startx + self.output_size[0],
                        starty: starty + self.output_size[1],
                        ]

        if is_video:
            image = torch.from_numpy(image.astype(np.float32))  # [T, H, W, C]
            if not self.no_mask:
                label = torch.from_numpy(label)
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # [1, H, W, C]
            if not self.no_mask:
                label = torch.from_numpy(label).unsqueeze(0)

        if not is_video:
            image = image[0]  # [H, W, C]
            label = label[0] if label is not None else None

        if "mode_prompt" in sample:
            sample = {
                "image": image,
                "label": label.long() if not self.no_mask else None,
                "mode_prompt": mode_prompt,
            }
        else:
            sample = {
                "image": image,
                "label": label.long() if not self.no_mask else None,
            }

        return sample


class CenterCropGenerator(object):
    def __init__(self, output_size, no_mask=False):
        self.output_size = np.array(output_size)
        self.no_mask = no_mask

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        is_video = (image.ndim == 4)
        if "mode_prompt" in sample:
            mode_prompt = sample["mode_prompt"]
        if is_video:
            T, H, W, C = image.shape
            scale = self.output_size / min(H, W)
            image = zoom(image, (1, scale[0], scale[1], 1), order=1)
            if not self.no_mask:
                label = zoom(label, (1, scale[0], scale[1]), order=0)

            _, x, y, _ = image.shape
            startx = x // 2 - (self.output_size[0] // 2)
            starty = y // 2 - (self.output_size[1] // 2)
            image = image[:, startx:startx + self.output_size[0], starty:starty + self.output_size[1], :]
            if not self.no_mask:
                label = label[:, startx:startx + self.output_size[0], starty:starty + self.output_size[1]]

            image = torch.from_numpy(image.astype(np.float32))
            if not self.no_mask:
                label = torch.from_numpy(label).long()

        else:
            H, W, C = image.shape
            scale = self.output_size / min(H, W)
            image = zoom(image, (scale[0], scale[1], 1), order=1)
            if not self.no_mask:
                label = zoom(label, (scale[0], scale[1]), order=0)

            x, y, _ = image.shape
            startx = x // 2 - (self.output_size[0] // 2)
            starty = y // 2 - (self.output_size[1] // 2)
            image = image[startx:startx + self.output_size[0], starty:starty + self.output_size[1], :]
            if not self.no_mask:
                label = label[startx:startx + self.output_size[0], starty:starty + self.output_size[1]]

            image = torch.from_numpy(image.astype(np.float32))
            if not self.no_mask:
                label = torch.from_numpy(label).long()
        if "mode_prompt" in sample:
            sample = {"image": image, "label": label.long(), "mode_prompt": mode_prompt}
        else:
            sample = {"image": image, "label": label.long()}
        return sample


class USdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False, video_seg=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.prompt = prompt
        self.video_seg = video_seg or "videos segmentation" in self.data_dir
        self.sample_list = []
        if not self.video_seg and "videos cls" not in self.data_dir:
            self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
            self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        if self.video_seg:
            self.data_dir = self.data_dir.replace("\\", "/")
            video_dir = os.path.join(self.data_dir, "videos", self.split)
            video_dir = video_dir.replace("\\", "/")
            datasetname = self.data_dir.strip("/").split("/")[-1]
            self.sample_list += [f"videos segmentation/{datasetname}/videos/{self.split}/{f}"
                                 for f in os.listdir(video_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip("\n")
        img_task = "img"
        img_path = os.path.join(self.data_dir, "imgs", img_name)
        img_path = img_path.replace("\\", "/")
        label_path = os.path.join(self.data_dir, "masks", img_name)
        label_path = label_path.replace("\\", "/")
        if not os.path.exists(label_path):
            label_path = label_path.replace("jpg", "png")

        if self.video_seg:
            img_task = "video"
            video_dir = os.path.join(self.data_dir, "annotations", self.split)
            img_name = img_name.split("/")[-1].split(".")[0]
            anno_path = os.path.join(video_dir, img_name + ".npz")
            anno_path = anno_path.replace("\\", "/")
            img_dir = os.path.join(self.data_dir, "videos", self.split)
            img_path = os.path.join(img_dir, img_name + ".npy")
            img_path = img_path.replace("\\", "/")
            frames, masks = load_npy_and_npz(img_path, anno_path, 2 if os.path.basename(self.data_dir) == "EchoNet" else 10)
            frames = frames.transpose((0, 2, 3, 1))
            label_info = open(
                os.path.join(self.data_dir, "config.yaml")).readlines()
            label_info_list = [info.strip().split(":") for info in label_info]
            for single_label_info in label_info_list:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                masks[masks == label_value_in_image] = 0 if label_index == 0 else 1
            sample = {"image": frames / 255.0, "label": masks}
            if self.prompt:
                sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
        else:
            image = cv2.imread(img_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if "HMC-QU" in self.data_dir:
            image = cv2.resize(image, (256, 256))
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

        if img_task == "img":
            label_info_list = [info.strip().split(":") for info in self.label_info]
            for single_label_info in label_info_list:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                label[label == label_value_in_image] = label_index
            sample = {"image": image / 255.0, "label": label}

        if self.prompt:
            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]

        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = img_name
        if self.prompt:
            if img_task == "img":
                dataset_name = img_path.split("/")[-3]
            if self.video_seg:
                dataset_name = img_path.split("/")[-4]
            sample["type_prompt"] = type_prompt_one_hot_dict[type_prompt_dict[dataset_name]]
            sample["position_prompt"] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample


class USdatasetCls(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()

        # BUSI
        self.sample_list = [sample for sample in self.sample_list if not "normal" in sample]

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip("\n")
        img_name = img_name.replace("\\", "/")
        img_path = os.path.join(self.data_dir, img_name)
        img_path = img_path.replace("\\", "/")

        image = cv2.imread(img_path)
        label = int(img_name.split("/")[0])

        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
        if self.transform:
            sample = self.transform(sample)
        sample["label"] = torch.from_numpy(np.array(label))
        sample["case_name"] = self.sample_list[idx].strip("\n")
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
            sample["type_prompt"] = type_prompt_one_hot_dict[type_prompt_dict[dataset_name]]
            sample["position_prompt"] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample
