import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from typing import Sequence

# prompt info dict
# task prompt
position_prompt_list = [
    "segmentation",
    "classification",
    "videos segmentation",
    "videos classification",
]
# position prompt
position_prompt_dict = {
    "MMOTU": "ovary",
    "BUSI": "breast",
    "TN3K": "thyroid",
    "CUBS": "carotid",
    "BUS-BRA": "breast",
    "Appendix": "appendix",
    "Fatty-Liver": "liver",
    "UDIAT": "breast",
    "DDTI": "thyroid",
    "HMC-QU": "cardiac",
    "Fetal_HC": "head",
    "BUSIS": "breast",
    "CCAU": "carotid",
    "kidneyUS_capsule": "kidney",
    "TG3K": "thyroid",
    "EchoNet-Dynamic": "cardiac",
    "EchoNet": "cardiac",
    "CAMUS": "cardiac",
    "Seven_Breast_Full": "breast",
    "Seven_Fetal_head_Full": "head",
    "Seven_Heart_Full": "cardiac",
    "Seven_Kidney_Full": "kidney",
    "Seven_Thyroid_Full": "thyroid",
    "VSegCA": "carotid",
    "VSegIJV": "carotid",
    "ZhangCA": "carotid",
    "JH_HC": "head",
    "Macao1": "head",
    "LiverCirrhosisCui": "liver",
    "LiverCirrhosisYang": "liver",
    "Breast": "breast",
    "Liver": "liver",
    "Prostate": "indis",
    "Thyroid": "thyroid",
    "Thyroid_1": "thyroid",
}

# type prompt
type_prompt_dict = {
    "DDTI": "tumor",
    "MMOTU": "tumor",
    "BUSI": "tumor",
    "HMC-QU": "organ",
    "TN3K": "tumor",
    "Fetal_HC": "organ",
    "BUSIS": "tumor",
    "CCAU": "organ",
    "CUBS": "organ",
    "BUS-BRA": "tumor",
    "kidneyUS_capsule": "organ",
    "TG3K": "organ",
    "EchoNet-Dynamic": "organ",
    "UDIAT": "tumor",
    "Appendix": "organ",
    "Fatty-Liver": "organ",
    "EchoNet": "organ",
    "CAMUS": "organ",
    "Seven_Breast_Full": "tumor",
    "Seven_Fetal_head_Full": "organ",
    "Seven_Heart_Full": "organ",
    "Seven_Kidney_Full": "organ",
    "Seven_Thyroid_Full": "tumor",
    "VSegCA": "organ",
    "VSegIJV": "organ",
    "ZhangCA": "organ",
    "JH_HC": "organ",
    "Macao1": "organ",
    "LiverCirrhosisCui": "organ",
    "LiverCirrhosisYang": "organ",
    "Breast": "tumor",
    "Liver": "organ",
    "Prostate": "organ",
    "Thyroid": "tumor",
    "Thyroid_1": "tumor",
}

# mode prompt
available_mode_prompt_list = [
    "DDTI",
    "MMOTU",
    "BUSI",
    "HMC-QU",
    "TN3K",
    "Fetal_HC",
    "BUSIS",
    "CCAU",
    "CUBS",
    "BUS-BRA",
    "kidneyUS_capsule",
    "TG3K",
    "EchoNet-Dynamic",
    "UDIAT",
    "EchoNet",
    "CAMUS",
    "Seven_Breast_Full",
    "Seven_Fetal_head_Full",
    "Seven_Heart_Full",
    "Seven_Kidney_Full",
    "Seven_Thyroid_Full",
    "VSegCA",
    "VSegIJV",
    "ZhangCA",
]


# prompt one-hot
# organ prompt
position_prompt_one_hot_dict = {
    "ovary": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "breast": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "thyroid": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "carotid": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "liver": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "cardiac": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "head": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "kidney": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "leg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "indis": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


# task prompt

task_prompt_one_hot_dict = {
    "segmentation": [1, 0, 0, 0],
    "classification": [0, 1, 0, 0],
    "videos segmentation": [0, 0, 1, 0],
    "videos classification": [0, 0, 0, 1],
}

# type prompt
type_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}
# mode prompt
mode_prompt_one_hot_dict = {
    "whole": [1, 0, 0, 0],
    "local": [0, 1, 0, 0],
    "location": [0, 0, 1, 0],
    "full": [0, 0, 0, 1],
}


def list_add_prefix(txt_path, prefix_1, prefix_2):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    if prefix_2 is not None:
        return [os.path.join(prefix_1, prefix_2, line.strip("\n")) for line in lines]
    else:
        return [os.path.join(prefix_1, line.strip("\n")) for line in lines]


class WeightedRandomSamplerDDP(DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        data_set,
        weights: Sequence[float],
        num_replicas: int,
        rank: int,
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(num_samples)
            )
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got " "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank :: self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def get_localImg(image, mask):
    if mask.sum() == 0:
        return image, mask, False
    else:
        x, y, w, h = cv2.boundingRect(mask)
        length = max(w, h)
        local_image = image[y : y + length, x : x + length, :]
        local_mask = mask[y : y + length, x : x + length]
        return local_image, local_mask, True


def get_enhanceImg(image, mask):
    if mask.sum() == 0:
        return image, False
    else:
        # mask[mask > 0] = 255
        bin_mark = mask.copy()
        bin_mark[bin_mark > 0] = 1
        enhance_image = image + (np.expand_dims(bin_mark, axis=2) * 0.1).astype("uint8")
        return enhance_image, True


def get_sample_list(seg_dataset, cls_dataset, image_seg_root, image_cls_root, split):
    return_list = []
    for dataset in seg_dataset:
        dataset_return_list = list_add_prefix(
            os.path.join(image_seg_root, dataset, split + ".txt"), "segmentation/" + dataset, "imgs"
        )

        if dataset == "BUSI":
            dataset_return_list = [sample for sample in dataset_return_list if not "normal" in sample]

        return_list.extend(dataset_return_list)
    for dataset in cls_dataset:
        dataset_return_list = list_add_prefix(
            os.path.join(image_cls_root, dataset, split + ".txt"), "classification/" + dataset, None
        )

        if dataset == "BUSI":
            dataset_return_list = [sample for sample in dataset_return_list if not "normal" in sample]

        return_list.extend(dataset_return_list)
    return return_list


def resolve_video_cls_dir(video_cls_root, dataset):
    if os.path.isdir(os.path.join(video_cls_root, "0")):
        return video_cls_root
    named_dir = os.path.join(video_cls_root, dataset)
    if os.path.isdir(named_dir):
        return named_dir
    return os.path.join(video_cls_root, f"out_all_{dataset.lower()}")


def get_video_cls_sample_list(video_cls_dataset, video_cls_root, split):
    return_list = []
    for dataset in video_cls_dataset:
        dataset_dir = resolve_video_cls_dir(video_cls_root, dataset)
        split_list = os.path.join(dataset_dir, split + ".txt")
        if os.path.isfile(split_list):
            dataset_return_list = list_add_prefix(split_list, "videos classification/" + dataset, None)
        else:
            dataset_return_list = [f"videos classification/{dataset}/{label}/{split}/{name}"
                                   for label in ("0", "1")
                                   for name in sorted(os.listdir(os.path.join(dataset_dir, label, split)))
                                   if name.lower().endswith(".npy")]
        return_list.extend(dataset_return_list)
    return return_list


def get_subset_len(seg_dataset, cls_dataset, image_seg_root, image_cls_root, split):
    subset_len = []
    for dataset in seg_dataset:
        dataset_subset_len = len(
            list_add_prefix(os.path.join(image_seg_root, dataset, split + ".txt"), dataset, "imgs")
        )

        if dataset == "BUSI":
            dataset_return_list = list_add_prefix(
                os.path.join(image_seg_root, dataset, split + ".txt"), dataset, "imgs"
            )
            dataset_subset_len = len([sample for sample in dataset_return_list if not "normal" in sample])

        subset_len.append(dataset_subset_len)
    for dataset in cls_dataset:
        dataset_subset_len = len(
            list_add_prefix(os.path.join(image_cls_root, dataset, split + ".txt"), dataset, None)
        )

        if dataset == "BUSI":
            dataset_return_list = list_add_prefix(
                os.path.join(image_cls_root, dataset, split + ".txt"), dataset, None
            )
            dataset_subset_len = len([sample for sample in dataset_return_list if not "normal" in sample])

        subset_len.append(dataset_subset_len)
    return subset_len


def get_video_cls_subset_len(video_cls_dataset, video_cls_root, split):
    subset_len = []
    for dataset in video_cls_dataset:
        dataset_subset_len = len(get_video_cls_sample_list([dataset], video_cls_root, split))
        subset_len.append(dataset_subset_len)
    return subset_len


def load_npy_and_npz(img_path: str,
                     anno_path: str,
                     frame_length: int = 11):
    # load video
    video = np.load(img_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    kpts_list = np.load(anno_path, allow_pickle=True)
    # ef, edv, esv = kpts_list['ef'], kpts_list['edv'], kpts_list['esv']

    # Collect masks:
    idx_list = []
    masks = []
    mask_list = kpts_list['fnum_mask'].tolist()
    for kpt in kpts_list['fnum_mask'].tolist().keys():
        idx_list.append(int(kpt))
        masks.append(mask_list[kpt])

    # Swap if ED before ES:
    if idx_list[0] > idx_list[-1]:
        idx_list.reverse()
        masks.reverse()

    # compute step:
    x0, x1 = idx_list[-1], idx_list[0]
    if frame_length == -1:
        frame_length = x0 + 1

    step = min(x0, (x0 - x1) / (frame_length - 1))

    # select frames inds:
    frame_inds = [int(idx_list[0] + step * i) for i in range(frame_length)]

    # Collect frames:
    frames = []
    for i in range(frame_length):
        frames.append(video[frame_inds[i]])

    select_masks = []
    if len(masks) <= len(frame_inds):
        select_masks = masks
    else:
        for i in range(frame_length):
            select_masks.append(masks[frame_inds[i]])

    select_masks = np.asarray(select_masks)
    imgs = np.asarray(frames)


    return imgs, select_masks


def sample_video_frames(video: np.ndarray, frame_length: int):
    total_frames = video.shape[0]
    if frame_length <= 0 or total_frames <= frame_length:
        return video
    frame_inds = np.linspace(0, total_frames - 1, frame_length, dtype=int)
    return video[frame_inds]


class USdatasetOmni(Dataset):
    def __init__(
        self,
        base_dir,
        split,
        transform=None,
        prompt=False,
        seg_use_dataset=None,
        cls_use_dataset=None,
        seg_video_use_dataset=None,
        cls_video_use_dataset=None,
        video_seg_root=None,
        video_cls_root=None,
        image_seg_root=None,
        image_cls_root=None,
    ):
        self.data_dir = base_dir
        self.image_seg_root = image_seg_root or os.path.join(base_dir, "segmentation")
        self.image_cls_root = image_cls_root or os.path.join(base_dir, "classification")
        self.video_seg_root = video_seg_root or os.path.join(base_dir, "videos segmentation")
        self.video_cls_root = video_cls_root or os.path.join(base_dir, "video classification")
        self.split = split
        self.transform = transform
        self.prompt = prompt
        self.sample_list = []
        self.subset_len = []
        self.frame_length = 10
        self.seg_use_dataset = list(seg_use_dataset or [])
        self.cls_use_dataset = list(cls_use_dataset or [])
        self.seg_video_use_dataset = list(seg_video_use_dataset or [])
        self.cls_video_use_dataset = list(cls_video_use_dataset or [])

        self.sample_list = get_sample_list(self.seg_use_dataset, self.cls_use_dataset, self.image_seg_root, self.image_cls_root, split)
        self.subset_len = get_subset_len(self.seg_use_dataset, self.cls_use_dataset, self.image_seg_root, self.image_cls_root, split)
        for d in self.seg_video_use_dataset:
            video_dir = os.path.join(self.video_seg_root, d, "videos", self.split)
            video_dir = video_dir.replace("\\", "/")
            self.sample_list += [f"videos segmentation/{d}/videos/{self.split}/{f}"
                                 for f in os.listdir(video_dir) if f.endswith(".npy")]
        self.sample_list += get_video_cls_sample_list(self.cls_video_use_dataset, self.video_cls_root, split)
        self.subset_len += get_video_cls_subset_len(self.cls_video_use_dataset, self.video_cls_root, split)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip("\n")
        img_name = img_name.replace("\\", "/")
        img_path = os.path.join(self.data_dir, img_name)
        img_path = img_path.replace("\\", "/")
        img_task = self.sample_list[idx].strip("\n").split("/")[0]
        dataset_name = img_name.split("/")[1]
        if img_task == "videos segmentation":
            img_path = os.path.join(self.video_seg_root, dataset_name, "videos", self.split, os.path.basename(img_name))
            prefix = os.path.splitext(os.path.basename(img_name))[0]
            anno_path = os.path.join(
                self.video_seg_root, dataset_name, "annotations", self.split, prefix + ".npz"
            )
            frames, masks = load_npy_and_npz(img_path, anno_path, 2 if dataset_name == "EchoNet" else self.frame_length)
            frames = frames.transpose((0, 2, 3, 1))
            label_info = open(os.path.join(self.video_seg_root, dataset_name, "config.yaml")).readlines()
            label_info_list = [info.strip().split(":") for info in label_info]
            for single_label_info in label_info_list:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                masks[masks == label_value_in_image] = 0 if label_index == 0 else 1
            sample = {"image": frames / 255.0, "label": masks}
            if self.prompt:
                sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
        elif img_task == "videos classification":
            rel_path = "/".join(img_name.split("/")[2:])
            img_path = os.path.join(resolve_video_cls_dir(self.video_cls_root, dataset_name), rel_path)
            img_path = img_path.replace("\\", "/")
            frames = np.load(img_path, allow_pickle=True)
            if frames.ndim != 4:
                raise ValueError(f"Expected four-dimensional video: {img_path}: {frames.shape}")
            if frames.shape[0] in (1, 3) and frames.shape[-1] not in (1, 3):
                frames = frames.transpose(1, 2, 3, 0)
            frames = sample_video_frames(frames, self.frame_length)
            label = int(img_name.split("/")[2])
            sample = {
                "image": frames / 255.0,
                "label": np.zeros(frames.shape[:3], dtype=np.uint8),
            }
            if self.prompt:
                sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
        else:
            if img_task == "segmentation":
                img_path = os.path.join(self.image_seg_root, dataset_name, "imgs", "/".join(img_name.split("/")[3:]))
            elif img_task == "classification":
                img_path = os.path.join(self.image_cls_root, dataset_name, "/".join(img_name.split("/")[2:]))
            image = cv2.imread(img_path)

        if "HMC-QU" in img_name:
            image = cv2.resize(image, (256, 256))

        if img_task == "segmentation":
            label_path = os.path.join(self.image_seg_root, dataset_name, "masks", "/".join(img_name.split("/")[3:]))
            if not os.path.exists(label_path):
                label_path = label_path.replace("jpg", "png")
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if "HMC-QU" in img_name:
                label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

            label_info = open(os.path.join(self.image_seg_root, dataset_name, "config.yaml")).readlines()
            label_info_list = [info.strip().split(":") for info in label_info]
            for single_label_info in label_info_list:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                label[label == label_value_in_image] = label_index
            # --------------------------------------------------------------------#
            if not self.prompt:
                sample = {"image": image / 255.0, "label": label}
            else:
                if random.random() > 0.5:
                    image, label, notZeroMask = get_localImg(image, label)
                    sample = {"image": image / 255.0, "label": label}
                    if notZeroMask:
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["local"]
                    else:
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                else:
                    sample = {"image": image / 255.0, "label": label}
                    sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
            # --------------------------------------------------------------------#

        elif img_task == "classification":
            label = int(img_name.split("/")[-2])

        if img_task == "classification":
            # --------------------------------------------------------------------#
            if not self.prompt:
                sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
            else:
                if dataset_name in available_mode_prompt_list:
                    random_number = random.random()
                    mask_path = os.path.join(
                        self.image_seg_root,
                        "/".join([img_name.split("/")[1], "masks", img_name.split("/")[3]]),
                    )
                    mask_path = mask_path.replace(".jpg", ".png").replace(".JPG", ".png")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if random_number < 0.3:
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                    elif random_number < 0.6:
                        image, mask, notZeroMask = get_localImg(image, mask)
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        if notZeroMask:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["local"]
                        else:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                    else:
                        image, notZeroMask = get_enhanceImg(image, mask)
                        sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                        if notZeroMask:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["location"]
                        else:
                            sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
                else:
                    sample = {"image": image / 255.0, "label": np.zeros(image.shape[:2])}
                    sample["mode_prompt"] = mode_prompt_one_hot_dict["whole"]
            # --------------------------------------------------------------------#
        if self.transform:
            sample = self.transform(sample)

        # label
        if img_task == "segmentation":
            sample["1dLabel"] = torch.from_numpy(np.array(0)).float()
            sample["2dLabel"] = sample["label"].float()
        elif img_task == "classification":
            sample["1dLabel"] = torch.from_numpy(np.array(label)).float()
            sample["2dLabel"] = torch.from_numpy(np.zeros(sample["image"].shape[0:2])).float()
        elif img_task == "videos segmentation":
            sample["1dLabel"] = torch.from_numpy(np.array(0)).float()
            sample["2dLabel"] = sample["label"]  # [T, H, W]
        elif img_task == "videos classification":
            sample["1dLabel"] = torch.tensor(label).float()
            T, H, W, _ = sample["image"].shape
            sample["2dLabel"] = torch.zeros((T, H, W)).float()

        sample["case_name"] = self.sample_list[idx].strip("\n")
        sample["type_prompt"] = type_prompt_one_hot_dict[type_prompt_dict[dataset_name]]
        sample["position_prompt"] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        sample["task"] = img_task
        sample["task_prompt"] = task_prompt_one_hot_dict[img_task]
        sample["dataset_name"] = dataset_name

        return sample


def save_sample(sample, out_dir="./debug_save", prefix="sample"):
    os.makedirs(out_dir, exist_ok=True)

    images = sample["image"]
    labels = sample["label"]
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        labels = np.expand_dims(labels, axis=0)
    _, C, _, _ = images.shape
    if C != 3:
        images = images.transpose(0, 3, 1, 2)
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    T = images.shape[0]
    for t in range(T):
        # ---- image ----
        img = images[t].transpose(1, 2, 0)   # (C,H,W) -> (H,W,C)
        img = (img * 255).clip(0,255).astype("uint8")
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_frame{t:02d}.png"), img)
    T = labels.shape[0]
    for t in range(T):
        # ---- label ----
        lbl = labels[t]  # (H,W)
        lbl = (lbl * 255).clip(0, 255).astype("uint8")
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_mask{t:02d}.png"), lbl)

    print(f"Saved {T} frames to {out_dir}")
