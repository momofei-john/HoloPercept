# HoloPercept

HoloPercept is a prompt-conditioned framework for ultrasound image and video classification and segmentation. This repository provides the model, data loaders, training code, and evaluation code.

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
| `.vscode/launch.json` | VS Code run configurations |

## Environment

Use Python 3.11 with a CUDA-enabled PyTorch installation for training. Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The Swin-Tiny backbone weights are specified by `MODEL.PRETRAIN_CKPT`. Pass their location with `--opts MODEL.PRETRAIN_CKPT <checkpoint>` when starting training. A trained HoloPercept checkpoint is required for evaluation.

## Data organization

Dataset roots are supplied at runtime:

| Argument | Expected directory |
| --- | --- |
| `--image_seg_root` | Image segmentation datasets |
| `--image_cls_root` | Image classification datasets |
| `--video_seg_root` | Video segmentation datasets |
| `--video_cls_root` | Video classification datasets |

Each image segmentation dataset contains `imgs/`, `masks/`, `config.yaml`, and `train.txt`, `val.txt`, `test.txt`; its split lists contain paths relative to `imgs/`. Each image classification dataset contains class directories, `config.yaml`, and the three split lists, whose paths are relative to the dataset directory.

Each video segmentation dataset contains `videos/{train,val,test}/*.npy`, matching `annotations/{train,val,test}/*.npz`, and `config.yaml`. Annotation files provide `fnum_mask`. CAMUS samples use 10 frames; EchoNet samples use 2 annotated frames.

Video classification datasets support either split lists with entries such as `0/example.npy` or class directories of the form `{0,1}/{train,val,test}/*.npy`. Video arrays may use `(T,H,W,C)` or `(C,T,H,W)` layout.

The `--root_path` argument remains available when the four task directories are located under a common root as `segmentation/`, `classification/`, `videos segmentation/`, and `video classification/`. Data and pretrained weights are provided separately.

## Training

The following PowerShell example selects UDIAT and DDTI for image segmentation, Appendix and Fatty-Liver for image classification, CAMUS and EchoNet for video segmentation, and Breast and Thyroid for video classification. Replace the path placeholders with local paths.

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
  --video_seg_datasets CAMUS,EchoNet --video_cls_datasets Breast,Thyroid `
  --val_seg_datasets UDIAT,DDTI --val_cls_datasets Appendix,Fatty-Liver `
  --val_video_seg_datasets CAMUS,EchoNet --val_video_cls_datasets Breast,Thyroid `
  --train_stage joint --batch_size 1 --max_epochs 200 `
  --output_dir ./exp_out/holopercept `
  --opts MODEL.PRETRAIN_CKPT $backbone
```

Dataset flags accept comma-separated names. Use an empty string to disable a task group. Training writes `latest.pth`, `best_model.pth`, and logs to `--output_dir`.

## Evaluation

```powershell
python test.py `
  --image_seg_root $imageSeg --image_cls_root $imageCls `
  --video_seg_root $videoSeg --video_cls_root $videoCls `
  --seg_datasets UDIAT,DDTI --cls_datasets Appendix,Fatty-Liver `
  --video_seg_datasets CAMUS,EchoNet --video_cls_datasets Breast,Thyroid `
  --checkpoint ./exp_out/holopercept/best_model.pth `
  --output_dir ./exp_out/evaluation
```

Evaluation writes `test_summary.json` and `test_summary.csv` to `--output_dir`.

## VS Code

Open the repository folder, select the Python interpreter, and choose **HoloPercept: train** or **HoloPercept: test** in Run and Debug. The launch configurations prompt for dataset roots and the backbone checkpoint.

## Implementation

The network uses task, anatomical position, image type, and mode prompts. `train.py` supports joint training and task-specific stages through `--train_stage`. The training loop uses Adam with an iteration-based polynomial learning-rate schedule. `test.py` reports Dice, HD95, and ASD for segmentation; AUC and accuracy for classification; and model parameters, FLOPs, and inference speed.

## Attribution

Attribution notices for components derived from the Swin Transformer and Swin-Unet codebases are retained in the source files.
