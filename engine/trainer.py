import os
import cv2
import sys
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import DiceLoss
from data.dataset import USdataset, USdatasetCls
from data.omni_dataset import USdatasetOmni, resolve_video_cls_dir
from data.dataset import RandomGenerator, CenterCropGenerator
from sklearn.metrics import roc_auc_score
from utils import omni_seg_test, omni_video_seg_test


def parse_dataset_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def prompt_batch_to_tensor(batch_data, device):
    """Convert prompt batches to [B, C] float tensors."""
    if torch.is_tensor(batch_data):
        tensor = batch_data.float()
    elif isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and torch.is_tensor(batch_data[0]):
        tensor = torch.stack([item.float() for item in batch_data], dim=0)
        if tensor.dim() == 2 and tensor.shape[0] in {2, 4, 11} and tensor.shape[1] != tensor.shape[0]:
            tensor = tensor.transpose(0, 1).contiguous()
    else:
        tensor = torch.as_tensor(np.array(batch_data), dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def set_requires_grad_by_name(model, train_keywords):
    for name, param in model.named_parameters():
        param.requires_grad = any(keyword in name for keyword in train_keywords)


def configure_training_stage(model, stage):
    if stage == "joint":
        for _, param in model.named_parameters():
            param.requires_grad = True
        return

    if stage == "stage1_seg":
        frozen_keywords = [
            "layers_task_cls_up",
            "layers_task_cls_skip",
            "layers_task_cls_head",
            "norm_task_cls",
            "class_aware_neighbor",
        ]
        for name, param in model.named_parameters():
            param.requires_grad = not any(keyword in name for keyword in frozen_keywords)
        return

    if stage == "stage2_cls":
        frozen_keywords = [
            "layers_task_seg_up",
            "layers_task_seg_skip",
            "layers_task_seg_head",
            "layers_vout",
            "key_extractor",
            "value_encoder",
            "decoder",
            "mask_decoder",
        ]
        for name, param in model.named_parameters():
            param.requires_grad = not any(keyword in name for keyword in frozen_keywords)
        return

    raise ValueError(f"Unsupported training stage: {stage}")


def weight_base_init(nn_dataset):
    from data.omni_dataset import position_prompt_dict

    position_num_dict = {}
    seg_use_dataset_num = len(nn_dataset.seg_use_dataset)
    for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
        if position_prompt_dict[dataset_name] not in position_num_dict:
            position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]
        else:
            position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
    for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
        if position_prompt_dict[dataset_name] not in position_num_dict:
            position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[
                seg_use_dataset_num + dataset_index
                ]
        else:
            position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[
                seg_use_dataset_num + dataset_index
                ]
    cls_video_offset = seg_use_dataset_num + len(nn_dataset.cls_use_dataset)
    for dataset_index, dataset_name in enumerate(getattr(nn_dataset, "cls_video_use_dataset", [])):
        if position_prompt_dict[dataset_name] not in position_num_dict:
            position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[
                cls_video_offset + dataset_index
            ]
        else:
            position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[
                cls_video_offset + dataset_index
            ]

    position_weight_dict = {}
    for position in position_num_dict:
        position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])

    all_sample_weight_list = []
    for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
        all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[
            dataset_index
        ]
    for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
        all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[
            seg_use_dataset_num + dataset_index
            ]
    for dataset_index, dataset_name in enumerate(getattr(nn_dataset, "cls_video_use_dataset", [])):
        all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[
            cls_video_offset + dataset_index
        ]

    return all_sample_weight_list


def infer_video_cls_num_classes(video_cls_root, dataset_name, split="test"):
    dataset_dir = resolve_video_cls_dir(video_cls_root, dataset_name)
    split_list = os.path.join(dataset_dir, split + ".txt")
    if os.path.isfile(split_list):
        with open(split_list, "r", encoding="utf-8") as f:
            return len({line.strip().replace("\\", "/").split("/")[0]
                        for line in f if line.strip()})
    return len([name for name in os.listdir(dataset_dir)
                if name.isdigit() and os.path.isdir(os.path.join(dataset_dir, name, split))])


def reduce_video_cls_logits(logits):
    if torch.is_tensor(logits) and logits.dim() == 3:
        return logits.mean(dim=1)
    return logits


def to_binary_seg_target(target):
    return (target > 0).long()


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def update_checkpoint_alias(target_path, alias_path):
    """Keep a plain file copy as the stable alias on Windows."""
    import shutil

    shutil.copy2(target_path, alias_path)


def train_holopercept(args, model, snapshot_path):
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- Logging ----
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ---- Hyperparams ----
    base_lr = args.base_lr
    batch_size = args.batch_size

    # ---- Dataset ----
    stage = getattr(args, "train_stage", "joint")
    seg_datasets = parse_dataset_arg(getattr(args, "seg_datasets", ""))
    cls_datasets = parse_dataset_arg(getattr(args, "cls_datasets", ""))
    video_seg_datasets = parse_dataset_arg(getattr(args, "video_seg_datasets", ""))
    video_cls_datasets = parse_dataset_arg(getattr(args, "video_cls_datasets", ""))
    if stage == "stage1_seg":
        cls_datasets = []
        video_cls_datasets = []
    elif stage == "stage2_cls":
        seg_datasets = []
        video_seg_datasets = []

    effective_batch_size = batch_size
    variable_video_shape = bool(video_seg_datasets or video_cls_datasets)
    if variable_video_shape and batch_size != 1:
        logging.info(
            "%s includes videos with variable frame counts or tensor ranks; forcing batch_size=1 for safe collation.",
            stage,
        )
        effective_batch_size = 1

    db_train = USdatasetOmni(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        prompt=args.prompt,
        seg_use_dataset=seg_datasets,
        cls_use_dataset=cls_datasets,
        seg_video_use_dataset=video_seg_datasets,
        cls_video_use_dataset=video_cls_datasets,
        video_seg_root=getattr(args, "video_seg_root", None),
        video_cls_root=getattr(args, "video_cls_root", None),
        image_seg_root=getattr(args, "image_seg_root", None),
        image_cls_root=getattr(args, "image_cls_root", None),
    )


    trainloader = DataLoader(
        db_train,
        batch_size=effective_batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )

    # ---- Model ----
    configure_training_stage(model, getattr(args, "train_stage", "joint"))
    model.train()

    # ---- Loss & Optimizer ----
    seg_ce_loss = CrossEntropyLoss()
    seg_dice_loss = DiceLoss()
    cls_ce_loss = CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=base_lr)

    # ---- Resume ----
    resume_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_epoch = checkpoint["epoch"]

    # ---- Logging ----
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    best_performance = 0.0
    best_epoch = 0
    global_iter_num = 0

    total_iterations = len(trainloader)
    max_iterations = args.max_epochs * total_iterations
    logging.info(
        "{} batch size. {} iterations per epoch. {} max iterations ".format(
            effective_batch_size, total_iterations, max_iterations
        )
    )

    iterator = tqdm(range(resume_epoch, args.max_epochs), ncols=70)

    # ---- Training Loop ----
    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))
        for i_batch, sampled_batch in tqdm(enumerate(trainloader)):
            image_batch = sampled_batch["image"].to(device)
            task_batch = sampled_batch["task"]
            label_batch1D = sampled_batch["1dLabel"].to(device)
            label_batch2D = sampled_batch["2dLabel"].to(device)
            dataset_name = sampled_batch["dataset_name"]
            if args.prompt:
                position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
                task_prompt = prompt_batch_to_tensor(sampled_batch["task_prompt"], device)
                mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
                type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)
                (x_seg, x_cls) = model(
                    (image_batch, position_prompt, task_prompt, mode_prompt, type_prompt)
                )
            else:
                (x_seg, x_cls) = model(image_batch)

            # ---- Loss ----
            x_seg_select_index = [t == "segmentation" for t in task_batch]
            x_cls_select_index = [t == "classification" for t in task_batch]
            x_video_seg_select_index = [t == "videos segmentation" for t in task_batch]
            x_video_cls_select_index = [t == "videos classification" for t in task_batch]

            # seg
            if sum(x_seg_select_index) > 0:
                target_seg = to_binary_seg_target(label_batch2D[x_seg_select_index])
                loss_ce = seg_ce_loss(
                    x_seg[x_seg_select_index],
                    target_seg,
                )
                loss_dice = seg_dice_loss(
                    x_seg[x_seg_select_index],
                    target_seg,
                    softmax=True,
                )
                loss_seg = 0.4 * loss_ce + 0.6 * loss_dice
            else:
                loss_seg = 0

            # cls
            if sum(x_cls_select_index) > 0:
                loss_cls = cls_ce_loss(
                    x_cls[x_cls_select_index],
                    label_batch1D[x_cls_select_index].long(),
                )
            else:
                loss_cls = 0

            if sum(x_video_seg_select_index) > 0:
                if "EchoNet" in dataset_name:
                    if len(x_video_seg_select_index) != x_seg.shape[0]:
                        x_video_seg_select_index = x_video_seg_select_index[:x_seg.shape[0]]

                    if not torch.is_tensor(x_video_seg_select_index):
                        x_video_seg_select_index = torch.tensor(x_video_seg_select_index, dtype=torch.bool,
                                                                device=x_seg.device)
                    target = label_batch2D[x_video_seg_select_index]
                    target = target[:, [0, -1], :, :]
                    logits = x_seg[x_video_seg_select_index]
                    logits = logits[:, [0, -1], :, :, :]
                    logits = logits.view(-1, logits.shape[2], logits.shape[3], logits.shape[4])  # (B*2, C, H, W)
                    target = target.view(-1, target.shape[2], target.shape[3])
                    target = to_binary_seg_target(target)
                elif "CAMUS" in dataset_name:
                    target = label_batch2D[x_video_seg_select_index]
                    target = target.view(-1, target.shape[2], target.shape[3])
                    target = to_binary_seg_target(target)
                    logits = x_seg[x_video_seg_select_index]
                    logits = logits.view(-1, logits.shape[2], logits.shape[3], logits.shape[4])
                else:
                    target = label_batch2D[x_video_seg_select_index]
                    target = target.view(-1, target.shape[2], target.shape[3])
                    target = to_binary_seg_target(target)
                    logits = x_seg[x_video_seg_select_index]
                    logits = logits.view(-1, logits.shape[2], logits.shape[3], logits.shape[4])

                loss_ce = seg_ce_loss(
                    logits,
                    target,
                )
                loss_dice = seg_dice_loss(
                    logits,
                    target,
                    softmax=True,
                )
                loss_video_seg = 0.4 * loss_ce + 0.6 * loss_dice
            else:
                loss_video_seg = 0

            if sum(x_video_cls_select_index) > 0:
                video_cls_logits = reduce_video_cls_logits(x_cls[x_video_cls_select_index])
                loss_video_cls = cls_ce_loss(
                    video_cls_logits,
                    label_batch1D[x_video_cls_select_index].long(),
                )
            else:
                loss_video_cls = 0

            if stage == "stage1_seg":
                loss = loss_seg + loss_video_seg
            elif stage == "stage2_cls":
                loss = loss_cls + loss_video_cls
            else:
                loss = loss_seg + loss_cls + loss_video_seg + loss_video_cls

            # ---- Backprop ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - global_iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            global_iter_num += 1
            writer.add_scalar("info/lr", lr_, global_iter_num)
            writer.add_scalar("info/total_loss", loss, global_iter_num)
            logging.info("global iteration %d and loss : %f" % (global_iter_num, loss.item()))
        # ---- Validation ----
        torch.cuda.empty_cache()

        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_num,
        }
        save_latest_path = os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num))
        if os.path.exists(os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num - 1))):
            os.remove(os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num - 1)))
            if os.path.exists(os.path.join(snapshot_path, "latest.pth")):
                os.remove(os.path.join(snapshot_path, "latest.pth"))
        torch.save(save_dict, save_latest_path)
        update_checkpoint_alias(save_latest_path, os.path.join(snapshot_path, "latest.pth"))

        model.eval()
        total_performance = 0.0

        # ================= Segmentation Validation =================
        seg_val_set = parse_dataset_arg(getattr(args, "val_seg_datasets", getattr(args, "seg_datasets", "")))
        if stage == "stage2_cls":
            seg_val_set = []

        seg_avg_performance = 0.0
        for dataset_name in seg_val_set:
            num_classes = (
                open(os.path.join(args.image_seg_root or os.path.join(args.root_path, "segmentation"), dataset_name, "config.yaml")).read().count("\n")
            )
            db_val = USdataset(
                base_dir=os.path.join(args.image_seg_root or os.path.join(args.root_path, "segmentation"), dataset_name),
                split="val",
                list_dir=os.path.join(args.image_seg_root or os.path.join(args.root_path, "segmentation"), dataset_name),
                transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt,
            )
            val_loader = DataLoader(
                db_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            logging.info("{} val iterations per epoch".format(len(val_loader)))

            metric_list = 0.0
            count_matrix = np.ones((len(db_val) * num_classes, num_classes - 1))
            for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                image, label = sampled_batch["image"], sampled_batch["label"]
                if args.prompt:
                    position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
                    task_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * position_prompt.shape[0],
                                               dtype=torch.float32, device=device)
                    mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
                    type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)

                    metric_i = omni_seg_test(
                        image, label, model, classes=num_classes, prompt=args.prompt,
                        position_prompt=position_prompt, task_prompt=task_prompt,
                        mode_prompt=mode_prompt, type_prompt=type_prompt,
                    )
                else:
                    metric_i = omni_seg_test(image, label, model, classes=num_classes)

                for sample_index in range(len(metric_i)):
                    if not metric_i[sample_index][1]:
                        count_matrix[
                            (i_batch * batch_size + sample_index) // (num_classes - 1),
                            sample_index % (num_classes - 1),
                        ] = 0
                metric_i = [element[0] for element in metric_i]
                metric_list += np.array(metric_i).sum()

            performance = metric_list / (count_matrix.sum() + 1e-6)
            writer.add_scalar("info/val_seg_metric_{}".format(dataset_name), performance, epoch_num)
            seg_avg_performance += performance

        if len(seg_val_set) > 0:
            seg_avg_performance = seg_avg_performance / (len(seg_val_set) + 1e-6)
            total_performance += seg_avg_performance
            writer.add_scalar("info/val_metric_seg_Total", seg_avg_performance, epoch_num)

        # ================= Classification Validation =================
        cls_val_set = parse_dataset_arg(getattr(args, "val_cls_datasets", getattr(args, "cls_datasets", "")))
        if stage == "stage1_seg":
            cls_val_set = []
        cls_avg_performance = 0.0

        for dataset_name in cls_val_set:
            num_classes = (
                open(os.path.join(args.image_cls_root or os.path.join(args.root_path, "classification"), dataset_name, "config.yaml")).read().count("\n")
            )
            db_val = USdatasetCls(
                base_dir=os.path.join(args.image_cls_root or os.path.join(args.root_path, "classification"), dataset_name),
                split="val",
                list_dir=os.path.join(args.image_cls_root or os.path.join(args.root_path, "classification"), dataset_name),
                transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt,
            )

            val_loader = DataLoader(
                db_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            logging.info("{} val iterations per epoch".format(len(val_loader)))
            model.eval()

            label_list = []
            prediction_prob_list = []
            for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                image, label = sampled_batch["image"], sampled_batch["label"]
                if args.prompt:
                    position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
                    task_prompt = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * position_prompt.shape[0],
                                               dtype=torch.float32, device=device)
                    mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
                    type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)

                    with torch.no_grad():
                        output = model(
                            (image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                             mode_prompt.cuda(), type_prompt.cuda())
                        )[1]
                else:
                    with torch.no_grad():
                        output = model(image.cuda())[1]

                out_label_back_transform = torch.cat([output[:, 0:1], output[:, 1:num_classes]], axis=1)
                output_prob = torch.softmax(out_label_back_transform, dim=1).data.cpu().numpy()

                label_list.append(label.numpy())
                prediction_prob_list.append(output_prob)

            label_list = np.expand_dims(
                np.concatenate((np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1
            ).astype("uint8")
            label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
            performance = roc_auc_score(
                label_list_OneHot,
                np.concatenate(
                    (np.array(prediction_prob_list[:-1]).reshape(-1, num_classes), prediction_prob_list[-1])),
                multi_class="ovo",
            )
            writer.add_scalar("info/val_cls_metric_{}".format(dataset_name), performance, epoch_num)
            cls_avg_performance += performance

        if len(cls_val_set) > 0:
            cls_avg_performance = cls_avg_performance / (len(cls_val_set) + 1e-6)
            total_performance += cls_avg_performance
            writer.add_scalar("info/val_metric_cls_Total", cls_avg_performance, epoch_num)

        # ================== Video Segmentation Validation =================
        video_seg_val_set = parse_dataset_arg(
            getattr(args, "val_video_seg_datasets", getattr(args, "video_seg_datasets", ""))
        )
        if stage == "stage2_cls":
            video_seg_val_set = []
        video_seg_avg_performance = 0.0
        for dataset_name in video_seg_val_set:
            num_classes = open(
                os.path.join(args.video_seg_root or os.path.join(args.root_path, "videos segmentation"), dataset_name, "config.yaml")
            ).read().count("\n")
            db_val = USdataset(
                base_dir=os.path.join(args.video_seg_root or os.path.join(args.root_path, "videos segmentation"), dataset_name),
                split="val",
                list_dir="",
                transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt,
                video_seg=True,
            )

            val_loader = DataLoader(
                db_val,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
            logging.info("{} video val iterations per epoch".format(len(val_loader)))
            metric_list = 0.0
            count_matrix = np.ones((len(db_val), max(num_classes - 1, 1)))

            for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                video, label = sampled_batch["image"], sampled_batch["label"]
                if args.prompt:
                    position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
                    task_prompt = torch.tensor(
                        [[0.0, 0.0, 1.0, 0.0]] * position_prompt.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
                    type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)

                    metric_i = omni_video_seg_test(
                        video,
                        label,
                        model,
                        classes=num_classes,
                        prompt=args.prompt,
                        position_prompt=position_prompt,
                        task_prompt=task_prompt,
                        mode_prompt=mode_prompt,
                        type_prompt=type_prompt,
                    )
                else:
                    metric_i = omni_video_seg_test(video, label, model, classes=num_classes)

                for sample_index in range(len(metric_i)):
                    if not metric_i[sample_index][1]:
                        count_matrix[
                            (i_batch * batch_size + sample_index) // max(num_classes - 1, 1),
                            sample_index % max(num_classes - 1, 1),
                        ] = 0
                metric_i = [element[0] for element in metric_i]
                metric_list += np.array(metric_i).sum()

            performance = metric_list / (count_matrix.sum() + 1e-6)
            writer.add_scalar("info/val_video_seg_metric_{}".format(dataset_name), performance, epoch_num)
            video_seg_avg_performance += performance

        if len(video_seg_val_set) > 0:
            video_seg_avg_performance = video_seg_avg_performance / (len(video_seg_val_set) + 1e-6)
            total_performance += video_seg_avg_performance
            writer.add_scalar("info/val_metric_video_seg_Total", video_seg_avg_performance, epoch_num)

        # ================= Video Classification Validation =================
        video_cls_val_set = parse_dataset_arg(
            getattr(args, "val_video_cls_datasets", getattr(args, "video_cls_datasets", ""))
        )
        if stage == "stage1_seg":
            video_cls_val_set = []
        video_cls_avg_performance = 0.0
        for dataset_name in video_cls_val_set:
            num_classes = infer_video_cls_num_classes(args.video_cls_root or os.path.join(args.root_path, "video classification"), dataset_name, split="val")
            db_val = USdatasetOmni(
                base_dir=args.root_path,
                split="val",
                transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt,
                seg_use_dataset=[],
                cls_use_dataset=[],
                seg_video_use_dataset=[],
                cls_video_use_dataset=[dataset_name],
                video_cls_root=getattr(args, "video_cls_root", None),
            )

            val_loader = DataLoader(
                db_val,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
            logging.info("{} video cls val iterations per epoch".format(len(val_loader)))
            model.eval()

            label_list = []
            prediction_prob_list = []
            for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                image, label = sampled_batch["image"], sampled_batch["1dLabel"]
                if args.prompt:
                    position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
                    task_prompt = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * position_prompt.shape[0],
                                               dtype=torch.float32, device=device)
                    mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
                    type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)

                    with torch.no_grad():
                        output = model(
                            (image.cuda(), position_prompt, task_prompt, mode_prompt, type_prompt)
                        )[1]
                else:
                    with torch.no_grad():
                        output = model(image.cuda())[1]

                output = reduce_video_cls_logits(output)
                out_label_back_transform = torch.cat([output[:, 0:1], output[:, 1:num_classes]], axis=1)
                output_prob = torch.softmax(out_label_back_transform, dim=1).data.cpu().numpy()

                label_list.append(label.numpy())
                prediction_prob_list.append(output_prob)

            labels_flat = np.concatenate([np.asarray(x).reshape(-1) for x in label_list]).astype("uint8")
            probs_flat = np.concatenate(prediction_prob_list, axis=0)
            if num_classes == 2:
                performance = roc_auc_score(labels_flat, probs_flat[:, 1])
            else:
                label_one_hot = np.eye(num_classes)[labels_flat]
                performance = roc_auc_score(label_one_hot, probs_flat, multi_class="ovo")
            writer.add_scalar("info/val_video_cls_metric_{}".format(dataset_name), performance, epoch_num)
            video_cls_avg_performance += performance

        if len(video_cls_val_set) > 0:
            video_cls_avg_performance = video_cls_avg_performance / (len(video_cls_val_set) + 1e-6)
            total_performance += video_cls_avg_performance
            writer.add_scalar("info/val_metric_video_cls_Total", video_cls_avg_performance, epoch_num)

        # ================= Save Best Model =================
        active_metric_groups = (
            int(len(seg_val_set) > 0)
            + int(len(cls_val_set) > 0)
            + int(len(video_seg_val_set) > 0)
            + int(len(video_cls_val_set) > 0)
        )
        TotalAvgPerformance = total_performance / max(active_metric_groups, 1)
        logging.info("This epoch %d Validation performance: %f" % (epoch_num, TotalAvgPerformance))
        logging.info("Best epoch so far: %d with performance: %f" % (best_epoch, best_performance))
        writer.add_scalar("info/val_metric_TotalMean", TotalAvgPerformance, epoch_num)

        if TotalAvgPerformance >= best_performance:
            best_epoch = epoch_num
            best_performance = TotalAvgPerformance
            logging.info("New best model at epoch %d with performance %f" % (epoch_num, TotalAvgPerformance))
            save_model_path = os.path.join(snapshot_path,
                                           "best_model_{}_{}.pth".format(epoch_num, round(best_performance, 4)))
            if os.path.exists(os.path.join(snapshot_path, "best_model.pth")):
                os.remove(os.path.join(snapshot_path, "best_model.pth"))
            torch.save(model.state_dict(), save_model_path)
            update_checkpoint_alias(save_model_path, os.path.join(snapshot_path, "best_model.pth"))
            logging.info("Saved best model to {}".format(save_model_path))

        model.train()

    writer.close()
    return "Training Finished!"


omni_train = train_holopercept
