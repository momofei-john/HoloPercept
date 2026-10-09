import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model.HoloPercept import HoloPercept
from engine.trainer import train_holopercept
from config.config import get_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="./datasets",
    help="root dir for data",
)
parser.add_argument("--image_seg_root", type=str, default=None,
                    help="directory containing image segmentation datasets")
parser.add_argument("--image_cls_root", type=str, default=None,
                    help="directory containing image classification datasets")
parser.add_argument("--video_seg_root", type=str, default=None,
                    help="directory containing CAMUS and EchoNet")
parser.add_argument("--video_cls_root", type=str, default=None,
                    help="directory containing video classification datasets")
parser.add_argument("--output_dir", type=str, default="./exp_out/holopercept", help="output dir")
parser.add_argument("--max_epochs", type=int, default=200, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=1e-4, help="segmentation network learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--seed", type=int, default=2024, help="random seed")
parser.add_argument(
    "--cfg",
    type=str,
    default="./config/configs/swin_tiny_patch4_window7_224_lite.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into non-overlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint",
    action="store_true",
    help="whether to use gradient checkpointing to save memory",
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

parser.add_argument("--pretrain_ckpt", type=str, help="pretrained checkpoint")

parser.add_argument("--prompt", default=True, action="store_true", help="using prompt for training")
parser.add_argument(
    "--prompt_ablation",
    type=str,
    default="full",
    choices=["full", "wo_task", "wo_mode", "wo_pos", "wo_type", "concat"],
    help="Prompt ablation variant for training.",
)

parser.add_argument("--num_workers", type=int, default=24)

parser.add_argument("--adapter_ft", action="store_true", help="using adapter for fine-tuning")

parser.add_argument(
    "--train_stage",
    type=str,
    default="joint",
    choices=["joint", "stage1_seg", "stage2_cls"],
    help="training strategy stage: joint, segmentation-first, or classification-second",
)
parser.add_argument(
    "--seg_datasets",
    type=str,
    default="BUS-BRA,BUSIS,CAMUS,DDTI,Fetal_HC,kidneyUS_capsule,UDIAT,CCAU",
    help="comma-separated image segmentation datasets",
)
parser.add_argument(
    "--cls_datasets",
    type=str,
    default="Appendix,BUS-BRA,Fatty-Liver,UDIAT,CUBS",
    help="comma-separated classification datasets",
)
parser.add_argument(
    "--video_seg_datasets",
    type=str,
    default="CAMUS,EchoNet",
    help="comma-separated video segmentation datasets",
)
parser.add_argument(
    "--video_cls_datasets",
    type=str,
    default="Breast,Thyroid",
    help="comma-separated video classification datasets",
)
parser.add_argument(
    "--val_seg_datasets",
    type=str,
    default="BUS-BRA,BUSIS,CAMUS,DDTI,Fetal_HC,kidneyUS_capsule,UDIAT,CCAU",
    help="comma-separated image segmentation validation datasets",
)
parser.add_argument(
    "--val_cls_datasets",
    type=str,
    default="Appendix,BUS-BRA,Fatty-Liver,UDIAT,CUBS",
    help="comma-separated classification validation datasets",
)
parser.add_argument(
    "--val_video_seg_datasets",
    type=str,
    default="CAMUS,EchoNet",
    help="comma-separated video segmentation validation datasets",
)
parser.add_argument(
    "--val_video_cls_datasets",
    type=str,
    default="Breast,Thyroid",
    help="comma-separated video classification validation datasets",
)

parser.add_argument(
    "--local_rank", "--local-rank",
    dest="local_rank",
    type=int,
    default=0,
    help="(legacy) local rank for distributed training (accepts --local_rank or --local-rank)"
)
# ----------------------------------------------------------------------------------------


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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    net = HoloPercept(
        config,
        prompt=args.prompt,
        prompt_ablation=args.prompt_ablation,
    ).cuda()
    if args.pretrain_ckpt is not None:
        net.load_from_self(args.pretrain_ckpt)
    else:
        net.load_from(config)

    if args.prompt and args.adapter_ft:

        for name, param in net.named_parameters():
            if "prompt_proj_mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    train_holopercept(args, net, args.output_dir)
