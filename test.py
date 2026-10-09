import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from medpy import metric
from sklearn.metrics import accuracy_score, roc_auc_score
from thop import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import get_config
from data.dataset import CenterCropGenerator, USdataset, USdatasetCls
from data.omni_dataset import USdatasetOmni, resolve_video_cls_dir
from model.HoloPercept import HoloPercept


def prompt_batch_to_tensor(batch_data, device):
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


def parse_dataset_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_label_lines(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def num_classes_from_config(config_path: str) -> int:
    return len(read_label_lines(config_path))


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


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: List[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def estimate_classification_variation(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, seed: int, iters: int = 200):
    rng = np.random.default_rng(seed)
    auc_values = []
    acc_values = []
    n = len(labels)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        sample_labels = labels[idx]
        if np.unique(sample_labels).size < 2:
            continue
        auc_values.append(roc_auc_score(sample_labels, probs[idx]))
        acc_values.append(accuracy_score(sample_labels, preds[idx]))
    return safe_std(auc_values), safe_std(acc_values)


def calculate_binary_seg_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, bool]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0:
        return float("nan"), float("nan"), float("nan"), False
    if pred.sum() == 0:
        dice = metric.binary.dc(pred, gt) if gt.sum() == 0 else 0.0
        return float(dice), float("inf"), float("inf"), True
    dice = float(metric.binary.dc(pred, gt))
    hd95 = float(metric.binary.hd95(pred, gt))
    asd = float(metric.binary.asd(pred, gt))
    return dice, hd95, asd, True


def build_task_prompt(task_name: str, batch_size: int, device: torch.device):
    mapping = {
        "image_seg": [1.0, 0.0, 0.0, 0.0],
        "image_cls": [0.0, 1.0, 0.0, 0.0],
        "video_seg": [0.0, 0.0, 1.0, 0.0],
        "video_cls": [0.0, 0.0, 0.0, 1.0],
    }
    return torch.tensor([mapping[task_name]] * batch_size, dtype=torch.float32, device=device)


def infer_seg_logits(model, image, sampled_batch, device, prompt: bool, task_name: str):
    if prompt:
        position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
        task_prompt = build_task_prompt(task_name, position_prompt.shape[0], device)
        mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
        type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)
        logits = model((image.to(device), position_prompt, task_prompt, mode_prompt, type_prompt))[0]
    else:
        logits = model(image.to(device))[0]
    return logits


def seg_logits_to_prediction(logits: torch.Tensor) -> np.ndarray:
    if logits.dim() == 4:
        return torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
    if logits.dim() == 5:
        return torch.argmax(torch.softmax(logits, dim=2), dim=2).squeeze(0).cpu().numpy()
    raise ValueError(f"Unsupported segmentation logits shape: {tuple(logits.shape)}")


def infer_cls_logits(model, image, sampled_batch, device, prompt: bool, task_name: str):
    if prompt:
        position_prompt = prompt_batch_to_tensor(sampled_batch["position_prompt"], device)
        task_prompt = build_task_prompt(task_name, position_prompt.shape[0], device)
        mode_prompt = prompt_batch_to_tensor(sampled_batch["mode_prompt"], device)
        type_prompt = prompt_batch_to_tensor(sampled_batch["type_prompt"], device)
        logits = model((image.to(device), position_prompt, task_prompt, mode_prompt, type_prompt))[1]
    else:
        logits = model(image.to(device))[1]
    return logits


def collapse_foreground_metrics(pred: np.ndarray, label: np.ndarray):
    return calculate_binary_seg_metrics(pred > 0, label > 0)


def evaluate_image_segmentation(args, model, device, test_save_path=None):
    results = {}
    for dataset_name in parse_dataset_arg(args.seg_datasets):
        dataset_dir = os.path.join(args.image_seg_root or os.path.join(args.root_path, "segmentation"), dataset_name)
        config_path = os.path.join(dataset_dir, "config.yaml")
        num_classes = num_classes_from_config(config_path)
        db_test = USdataset(
            base_dir=dataset_dir,
            split="test",
            list_dir=dataset_dir,
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt,
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        dices, hd95s, asds = [], [], []
        model.eval()
        collapse_multiclass = num_classes > 2
        for sampled_batch in tqdm(testloader, desc=f"image_seg:{dataset_name}"):
            image = sampled_batch["image"]
            label = sampled_batch["label"].squeeze(0).cpu().numpy()
            case_name = sampled_batch["case_name"][0]
            with torch.no_grad():
                logits = infer_seg_logits(model, image, sampled_batch, device, args.prompt, "image_seg")
                pred = seg_logits_to_prediction(logits)
            dice, hd95, asd, valid = collapse_foreground_metrics(pred, label) if collapse_multiclass else calculate_binary_seg_metrics(pred == 1, label == 1)
            if valid:
                dices.append(dice)
                hd95s.append(hd95)
                asds.append(asd)
            logging.info("%s case %s DSC %.4f HD95 %.4f ASD %.4f", dataset_name, case_name, dice, hd95, asd)

        results[dataset_name] = {
            "num_cases": len(dices),
            "DSC_mean": safe_mean(dices),
            "DSC_std": safe_std(dices),
            "HD95_mean": safe_mean(hd95s),
            "HD95_std": safe_std(hd95s),
            "ASD_mean": safe_mean(asds),
            "ASD_std": safe_std(asds),
            "note": "foreground-vs-background evaluation" if collapse_multiclass else "",
        }
    return results


def evaluate_video_segmentation(args, model, device, test_save_path=None):
    results = {}
    for dataset_name in parse_dataset_arg(args.video_seg_datasets):
        db_test = USdataset(
            base_dir=os.path.join(args.video_seg_root or os.path.join(args.root_path, "videos segmentation"), dataset_name),
            split="test",
            list_dir="",
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt,
            video_seg=True,
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        dices, hd95s, asds = [], [], []
        model.eval()
        for sampled_batch in tqdm(testloader, desc=f"video_seg:{dataset_name}"):
            image = sampled_batch["image"]
            label = sampled_batch["label"].squeeze(0).cpu().numpy()
            case_name = sampled_batch["case_name"][0]
            with torch.no_grad():
                logits = infer_seg_logits(model, image, sampled_batch, device, args.prompt, "video_seg")
                pred = seg_logits_to_prediction(logits)
            for frame_idx in range(pred.shape[0]):
                dice, hd95, asd, valid = collapse_foreground_metrics(pred[frame_idx], label[frame_idx])
                if valid:
                    dices.append(dice)
                    hd95s.append(hd95)
                    asds.append(asd)
            logging.info("%s case %s frames %d", dataset_name, case_name, pred.shape[0])
        results[dataset_name] = {
            "num_frames": len(dices),
            "DSC_mean": safe_mean(dices),
            "DSC_std": safe_std(dices),
            "HD95_mean": safe_mean(hd95s),
            "HD95_std": safe_std(hd95s),
            "ASD_mean": safe_mean(asds),
            "ASD_std": safe_std(asds),
            "note": "foreground-vs-background frame-wise evaluation",
        }
    return results


def evaluate_image_classification(args, model, device):
    results = {}
    for dataset_name in parse_dataset_arg(args.cls_datasets):
        db_test = USdatasetCls(
            base_dir=os.path.join(args.image_cls_root or os.path.join(args.root_path, "classification"), dataset_name),
            split="test",
            list_dir=os.path.join(args.image_cls_root or os.path.join(args.root_path, "classification"), dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt,
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        labels, preds, probs = [], [], []
        model.eval()
        for sampled_batch in tqdm(testloader, desc=f"image_cls:{dataset_name}"):
            image = sampled_batch["image"]
            label = int(sampled_batch["label"].item())
            case_name = sampled_batch["case_name"][0]
            with torch.no_grad():
                logits = infer_cls_logits(model, image, sampled_batch, device, args.prompt, "image_cls")
                output_prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(output_prob))
            labels.append(label)
            preds.append(pred)
            probs.append(output_prob[1] if output_prob.shape[0] > 1 else output_prob[0])
            logging.info("%s case %s label %d pred %d prob1 %.4f", dataset_name, case_name, label, pred, probs[-1])
        labels = np.asarray(labels, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        auc_std, acc_std = estimate_classification_variation(labels, probs, preds, args.seed)
        results[dataset_name] = {
            "num_cases": int(labels.shape[0]),
            "AUC": float(roc_auc_score(labels, probs)),
            "AUC_std": auc_std,
            "Acc": float(accuracy_score(labels, preds)),
            "Acc_std": acc_std,
        }
    return results


def evaluate_video_classification(args, model, device):
    results = {}
    for dataset_name in parse_dataset_arg(args.video_cls_datasets):
        num_classes = infer_video_cls_num_classes(args.video_cls_root or os.path.join(args.root_path, "video classification"), dataset_name, split="test")
        db_test = USdatasetOmni(
            base_dir=args.root_path,
            split="test",
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt,
            seg_use_dataset=[],
            cls_use_dataset=[],
            seg_video_use_dataset=[],
            cls_video_use_dataset=[dataset_name],
            video_cls_root=args.video_cls_root,
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        labels, preds, probs = [], [], []
        model.eval()
        for sampled_batch in tqdm(testloader, desc=f"video_cls:{dataset_name}"):
            image = sampled_batch["image"]
            label = int(sampled_batch["1dLabel"].item())
            case_name = sampled_batch["case_name"][0]
            with torch.no_grad():
                logits = infer_cls_logits(model, image, sampled_batch, device, args.prompt, "video_cls")
                logits = reduce_video_cls_logits(logits)
                output_prob = torch.softmax(logits[:, :num_classes], dim=1).cpu().numpy()[0]
            pred = int(np.argmax(output_prob))
            labels.append(label)
            preds.append(pred)
            probs.append(output_prob[1] if num_classes == 2 else output_prob.tolist())
            logging.info("%s case %s label %d pred %d", dataset_name, case_name, label, pred)
        labels = np.asarray(labels, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.int64)
        if num_classes == 2:
            prob_arr = np.asarray(probs, dtype=np.float64)
            auc = float(roc_auc_score(labels, prob_arr))
            auc_std, acc_std = estimate_classification_variation(labels, prob_arr, preds, args.seed)
        else:
            prob_arr = np.asarray(probs, dtype=np.float64)
            auc = float(roc_auc_score(np.eye(num_classes)[labels], prob_arr, multi_class="ovo"))
            auc_std, acc_std = float("nan"), float("nan")
        results[dataset_name] = {
            "num_cases": int(labels.shape[0]),
            "AUC": auc,
            "AUC_std": auc_std,
            "Acc": float(accuracy_score(labels, preds)),
            "Acc_std": acc_std,
        }
    return results


def measure_model_stats(model, device, img_size: int, prompt: bool, warmup: int, iters: int):
    model.eval()
    dummy_image = torch.randn(1, img_size, img_size, 3, device=device)
    if prompt:
        position_prompt = torch.zeros(1, 11, device=device)
        position_prompt[:, 1] = 1
        task_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        mode_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        type_prompt = torch.tensor([[1.0, 0.0]], device=device)
        model_input = (dummy_image, position_prompt, task_prompt, mode_prompt, type_prompt)
    else:
        model_input = dummy_image
    params = sum(p.numel() for p in model.parameters())
    flops, _ = profile(model, inputs=(model_input,), verbose=False)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(model_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(model_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    fps = iters / elapsed
    return {
        "parameters": int(params),
        "flops": float(flops),
        "fps": float(fps),
        "latency_ms": float(1000.0 / fps),
    }


def load_ckpt_auto(snapshot, model, device="cpu", strict=False):
    ckpt = torch.load(snapshot, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model_keys = list(model.state_dict().keys())
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in state.keys())
    if model_has_module and not ckpt_has_module:
        new_state = {"module." + k: v for k, v in state.items()}
    elif (not model_has_module) and ckpt_has_module:
        new_state = {k[7:]: v for k, v in state.items()}
    else:
        new_state = state
    return model.load_state_dict(new_state, strict=strict)


def resolve_checkpoint(args):
    snapshot = args.checkpoint
    if snapshot:
        return snapshot
    best_model = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model):
        return best_model
    latest_model = os.path.join(args.output_dir, f"latest_{args.max_epochs - 1}.pth")
    if os.path.exists(latest_model):
        return latest_model
    return ""


def save_summary(output_dir: str, summary: Dict):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "test_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "test_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "dataset", "metric", "value"])
        for task_name in ["video_segmentation", "image_segmentation", "image_classification", "video_classification"]:
            for dataset, metrics in summary[task_name].items():
                for key, value in metrics.items():
                    writer.writerow([task_name, dataset, key, value])
        for key, value in summary["model_stats"].items():
            writer.writerow(["model", "global", key, value])


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./datasets", help="root dir for data")
parser.add_argument("--image_seg_root", type=str, default=None, help="directory containing image segmentation datasets")
parser.add_argument("--image_cls_root", type=str, default=None, help="directory containing image classification datasets")
parser.add_argument("--video_seg_root", type=str, default=None, help="directory containing CAMUS and EchoNet")
parser.add_argument("--video_cls_root", type=str, default=None, help="directory containing video classification datasets")
parser.add_argument("--output_dir", type=str, default="./exp_out/holopercept", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--is_saveout", default=True, action="store_true")
parser.add_argument("--test_save_dir", type=str, default="./predictions")
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cfg", type=str, default="config/configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE")
parser.add_argument("--opts", default=None, nargs="+")
parser.add_argument("--zip", action="store_true")
parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
parser.add_argument("--resume")
parser.add_argument("--accumulation-steps", type=int)
parser.add_argument("--use-checkpoint", action="store_true")
parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])
parser.add_argument("--tag")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--throughput", action="store_true")
parser.add_argument("--prompt", default=True, action="store_true")
parser.add_argument(
    "--prompt_ablation",
    type=str,
    default="full",
    choices=["full", "wo_task", "wo_mode", "wo_pos", "wo_type", "concat"],
    help="Prompt ablation variant used by the checkpoint.",
)
parser.add_argument("--checkpoint", type=str, default="", help="trained HoloPercept checkpoint; defaults to output_dir/best_model.pth when available")
parser.add_argument(
    "--seg_datasets",
    type=str,
    default="BUS-BRA,BUSIS,CAMUS,DDTI,Fetal_HC,kidneyUS_capsule,UDIAT,CCAU",
)
parser.add_argument(
    "--cls_datasets",
    type=str,
    default="Appendix,BUS-BRA,Fatty-Liver,UDIAT,CUBS",
)
parser.add_argument(
    "--video_seg_datasets",
    type=str,
    default="CAMUS,EchoNet",
)
parser.add_argument(
    "--video_cls_datasets",
    type=str,
    default="Breast,Thyroid",
)
parser.add_argument("--fps_warmup", type=int, default=20)
parser.add_argument("--fps_iters", type=int, default=100)
args = parser.parse_args()
config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = resolve_checkpoint(args)
    if not snapshot or not os.path.isfile(snapshot):
        parser.error("a trained checkpoint is required; provide --checkpoint or output_dir/best_model.pth")

    net = HoloPercept(config, prompt=args.prompt, prompt_ablation=args.prompt_ablation).to(device)
    if config.MODEL.PRETRAIN_CKPT and os.path.isfile(config.MODEL.PRETRAIN_CKPT):
        net.load_from(config)
    load_ckpt_auto(snapshot, net, device=device, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "test_result.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("checkpoint: %s", snapshot)

    if args.is_saveout:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(args.test_save_dir, exist_ok=True)
        test_save_path = args.test_save_dir
    else:
        test_save_path = None

    summary = {
        "checkpoint": snapshot,
        "prompt_ablation": args.prompt_ablation,
        "seed": args.seed,
        "video_segmentation": evaluate_video_segmentation(args, net, device, test_save_path),
        "image_segmentation": evaluate_image_segmentation(args, net, device, test_save_path),
        "image_classification": evaluate_image_classification(args, net, device),
        "video_classification": evaluate_video_classification(args, net, device),
        "model_stats": measure_model_stats(net, device, args.img_size, args.prompt, args.fps_warmup, args.fps_iters),
    }
    save_summary(args.output_dir, summary)
    logging.info(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
