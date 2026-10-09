# HoloPercept

This is the official source code repository for **“HoloPercept: A Dual-Prompt-Guided and Memory-Enhanced General Ultrasound Model for Image–Video Understanding.”**

## Project structure

| Path | Description |
| --- | --- |
| `model/` | HoloPercept network and supporting modules |
| `data/` | Image and video dataset loaders |
| `engine/trainer.py` | Training and validation |
| `config/` | Model configuration |
| `train.py` | Training entry point |
| `test.py` | Evaluation entry point |
| `utils.py` | Loss functions and metrics |

## Environment

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Download the pretrained [Swin-Tiny checkpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) and place it at the default path:

```text
pretrained_ckpt/swin_tiny_patch4_window7_224.pth
```

The path can be overridden with `--opts MODEL.PRETRAIN_CKPT <checkpoint>` when starting training. A trained HoloPercept checkpoint is required for evaluation.

## Data organization

## Training

The following PowerShell example selects UDIAT and DDTI for image segmentation, Appendix and Fatty-Liver for image classification, CAMUS and EchoNet for video segmentation, and Breast for video classification.

```powershell
$imageSeg = "<image-segmentation-root>"
$imageCls = "<image-classification-root>"
$videoSeg = "<video-segmentation-root>"
$videoCls = "<video-classification-root>"
$backbone = "<swin-tiny-checkpoint.pth>"

python train.py `
  --image_seg_root $imageSeg --image_cls_root $imageCls `
  --video_seg_root $videoSeg --video_cls_root $videoCls `
  --seg_datasets UDIAT,DDTI --cls_datasets Appendix,Fatty-Liver `
  --video_seg_datasets CAMUS,EchoNet --video_cls_datasets Breast `
  --val_seg_datasets UDIAT,DDTI --val_cls_datasets Appendix,Fatty-Liver `
  --val_video_seg_datasets CAMUS,EchoNet --val_video_cls_datasets Breast `
  --train_stage joint --batch_size 1 --max_epochs 200 `
  --output_dir ./exp_out/holopercept `
  --opts MODEL.PRETRAIN_CKPT $backbone
```

## Evaluation

```powershell
python test.py `
  --image_seg_root $imageSeg --image_cls_root $imageCls `
  --video_seg_root $videoSeg --video_cls_root $videoCls `
  --seg_datasets UDIAT,DDTI --cls_datasets Appendix,Fatty-Liver `
  --video_seg_datasets CAMUS,EchoNet --video_cls_datasets Breast `
  --checkpoint ./exp_out/holopercept/best_model.pth `
  --output_dir ./exp_out/evaluation
```

Evaluation writes `test_summary.json` and `test_summary.csv` to `--output_dir`.

This repository is built upon the [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) codebase. We thank the authors for making their work publicly available.
