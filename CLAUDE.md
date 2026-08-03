# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Google DeepMind's research repository for Tracking Any Point (TAP) — point tracking in video. It hosts multiple generations of models (TAP-Net → TAPIR → TAPNext → TAPNext++), benchmark datasets (TAP-Vid, TAPVid-3D, RoboTAP), training infrastructure, and inference demos.

## Installation & Setup

```bash
# Inference only (JAX)
pip install .

# Inference + PyTorch models (TAPIR, TAPNext, TAPNext++)
pip install ".[torch]"

# Training (adds TensorFlow, Kubric, etc.)
pip install ".[train]"

# TAPVid-3D evaluation
pip install ".[tapvid3d_eval]"
```

For CUDA support, install the correct JAX CUDA version first: https://github.com/jax-ml/jax#installation

**Training also requires system dependencies** (see `tapnet/training/README.md`):
```bash
sudo apt update && sudo apt install ffmpeg libopenexr-dev
```

## Running Inference

**Colab demos** (simplest way): notebooks in `colabs/`. Key ones:
- `colabs/torch_tapnextpp_demo.ipynb` — TAPNext++ (latest, best; PyTorch)
- `colabs/tapnext_demo.ipynb` — BootsTAPNext (JAX)
- `colabs/torch_tapnext_demo.ipynb` — BootsTAPNext (PyTorch re-implementation, same architecture & weights)
- `colabs/tapir_demo.ipynb` — Offline TAPIR/BootsTAPIR (JAX)
- `colabs/torch_tapir_demo.ipynb` — Offline TAPIR/BootsTAPIR (PyTorch)
- `colabs/causal_tapir_demo.ipynb` — Online (causal) TAPIR (JAX)
- `colabs/torch_causal_tapir_demo.ipynb` — Online (causal) TAPIR (PyTorch)
- `colabs/tapir_rainbow_demo.ipynb` — Rainbow visualization (foreground/background segmentation + camera motion correction)
- `colabs/tapir_clustering.ipynb` — RoboTAP segmentation algorithm
- `colabs/trajan_demo.ipynb` — TRAJAN autoencoder
- `colabs/optical_flow_track_assist.ipynb` — Optical flow tracking assistance
- `colabs/kubric_for_tapvid(3d).ipynb` — Groundtruth point track generation with Kubric

**Live webcam demo** (JAX, online TAPIR):
```bash
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
python3 ./tapnet/live_demo.py
```

**PyTorch live demo** (online TAPIR, PyTorch):
```bash
python3 ./tapnet/pytorch_live_demo.py
```
Uses `torch.load("tapnet/checkpoints/causal_bootstapir_checkpoint.pt")` — download the `.pt` checkpoint first.

## Checkpoints

Available on [HuggingFace](https://huggingface.co/google/tapnet). Downloaded automatically in colab demos or manually from GCS URLs in the README table. Checkpoint formats:
- JAX models: `.npy` (NumpyFileCheckpointer format — a dict with `params` and `state` keys)
- PyTorch models: `.pt` or `.ckpt`
- TAPNext JAX: `.npz`

## Architecture

### Pipeline

All models follow the same conceptual pipeline:
1. **Feature extraction**: ResNet (TAPIR) or TRecViT (TAPNext) backbone encodes video frames
2. **Query encoding**: Bilinear-interpolate features at query point locations on query frames
3. **Cost volume / correlation**: Cross-correlate query features with all frame features
4. **Localization**: Soft-argmax over cost volume → initial track + occlusion + expected distance
5. **Refinement** (TAPIR only): Iterative PIPs MLP-Mixer refines tracks at multiple resolutions
6. **Output**: `tracks [B,N,F,2]`, `occlusion [B,N,F]`, `expected_dist [B,N,F]`

### Model Generations

| Model | Framework | Backbone | Key Innovation |
|-------|-----------|----------|---------------|
| TAP-Net | JAX/Haiku | TSM-ResNet18 | Baseline cost-volume regression |
| TAPIR | JAX/Haiku + PyTorch | ResNet18 (+ optional ExtraConvs) | Two-stage: matching + iterative PIPs refinement |
| TAPNext | PyTorch | TrecViT-B | Next-token prediction with SSM (LRU) + ViT blocks |
| TAPNext++ | PyTorch | TrecViT-B | Long-term tracking (1024 frames), occlusion, re-detection |
| TRAJAN | Flax/Linen | Transformer | Point trajectory autoencoder (separate from tracking) |

TAPIR has **online (causal)** and **offline** variants. Online uses causal depthwise convolutions; offline sees the full video. TAPNext/TAPNext++ are inherently online (recurrent, frame-by-frame).

### Key Architectural Concepts

- **FeatureGrids**: NamedTuple of `(lowres, hires, resolutions)` — per-resolution feature pyramids from the backbone. `lowres` = 256-dim (ResNet unit_3), `hires` = 128-dim (ResNet unit_1).
- **QueryFeatures**: NamedTuple of `(lowres, hires, resolutions)` — features sampled at query point locations.
- **PIPs refinement** (TAPIR): Iterative MLP-Mixer with depthwise temporal convolutions refining position, occlusion, and expected distance. 4 iterations per resolution level.
- **Cost Volume**: `einsum('bnc,bthwc->tbnhw', query_feats, feature_grid)` — can be very memory-intensive, so queries are chunked with `query_chunk_size`.
- **Causal/Online mode**: Tracks are built incrementally. Each forward pass returns `causal_context` (hidden states of depthwise conv layers) to feed into the next frame.
- **Support points** (TAPNext++): Small grid of auxiliary points co-tracked with query points via shared attention, improving robustness.

### Coordinate Conventions

- **2D points**: `[x, y]` raster coordinates. (0,0) = upper-left of upper-left pixel; (w, h) = lower-right of lower-right pixel.
- **Query points**: `[t, y, x]` — frame coordinate `t` plus raster `y, x`. `t` = 0 is first frame; `t` = 0.5 is halfway between frames 0 and 1.
- **3D points**: `[t, y, x]` order, where `t` is fractional frame index.
- **Storage datasets**: Coordinates are in normalized raster coords [(0,0), (1,1)], but code immediately converts to regular raster coords.
- **Caveat**: Some internal code paths use `[y, x]` ordering (e.g., positional encoding in `tapir_model.py:216`, coordinate grids in `tapnext_torch.py:33`). The public API is always `[x, y]`, but when working with internals, double-check the coordinate order.

## Repository Structure

```
tapnet/
├── models/          # JAX/Haiku: TAPIR, TAP-Net, ResNet, TSM-ResNet, SSM ViT
├── torch/           # PyTorch: TAPIR re-implementation + ResNet + utilities
├── tapnext/         # TAPNext PyTorch: TRecViT, LRU/SSM modules, losses, parallel scan
├── tapnextpp/       # TAPNext++ improvements, metrics (AJ_RD), VOTSp2026 tracker
│   ├── augmentations/   # Homography and roll augmentations for training
│   ├── metrics/         # AJ_RD (Average Jaccard with Re-Detection) metric
│   └── votsp2026/       # VOT Sp 2026 challenge submission tracker
├── trajan/          # TRAJAN: point trajectory autoencoder (Flax/Linen)
├── training/        # JAX/Jaxline training loop (experiment, task, loss)
├── tapvid/          # TAP-Vid benchmark: dataset loaders, metrics (Jaccard, Pts Within Thresh, Occ Acc)
├── tapvid3d/        # TAPVid-3D benchmark: metrics, splits, annotation generation
├── robotap/         # RoboTAP point-track-based clustering
├── utils/           # Shared utilities: losses, transforms, model_utils, viz, optimizers
├── configs/         # JAX training configs (tapir, causal_tapir, tapir_bootstrap, tapnet)
└── colabs/          # Demo notebooks (see Running Inference above)
```

## Training (JAX)

Training uses the Jaxline framework with Haiku for model definition and the Kubric synthetic dataset. Not intended to be lightweight — requires TPU or multi-GPU setup.

Key files:
- `tapnet/training/experiment.py` — Jaxline experiment, data loading, pmapped update loop, checkpointing
- `tapnet/training/supervised_point_prediction.py` — Task: forward pass, Huber + occlusion + expected-dist loss
- `tapnet/training/task.py` — Abstract task interface
- `configs/tapir_config.py` — Example config with optimizer, dataset, eval settings
- `tapnet/utils/model_utils.py` — `tapnet_loss()` (Huber + BCE + expected dist), `huber_loss()`, `heatmaps_to_points()`

Training entry point: `python -m tapnet.training.experiment --config=configs/tapir_config.py`

## TAPNext++ VOTSp2026 Standalone API

The VOTSp2026 submission (`tapnet/tapnextpp/votsp2026/`) contains a reusable `TAPNextPP` wrapper (`model.py`) that can be used outside the VOT toolkit for frame-by-frame online tracking:

```python
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

model = TAPNextPP.from_checkpoint("tapnextpp_512.ckpt", device="cuda", input_resolution=512)
positions, visible, state = model.track_frame(frame_bgr, query_points_xy=np.array([[x, y]]))
for frame in subsequent_frames:
    positions, visible, state = model.track_frame(frame, state=state)
```

This model wraps TAPNext and adds support points (64 local grid points co-tracked with query points via shared attention, then discarded). The 512×512 checkpoint is downloaded automatically from `https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt` (cached to `~/.cache/tapnextpp/`). When building on top of `TAPNextPP`, note it operates in BGR input space and display-space output coordinates; coordinate transforms are handled by `utils.py` in the same directory.

## Testing & Linting

This is a research codebase — **there is no test suite, no CI, and no linting configuration**. The `pyproject.toml` defines package metadata only. When making changes, rely on colab demos for smoke-testing inference, or run a single forward pass to verify shapes.

## Key Dependencies

- **JAX stack**: `jax`, `dm-haiku`, `optax`, `chex`, `jaxline`
- **PyTorch stack**: `torch`, `torchvision` (optional: `pip install ".[torch]"`)
- **Training extras**: `tensorflow`, `tensorflow-datasets`, `kubric`
- **Media**: `mediapy` (video I/O), `opencv-python` (webcam), `matplotlib`
- **Manipulation**: `einops`, `einshape`

## Notes

- **GCS buckets**: Standard checkpoints live in `gs://dm-tapnet/`; the TAPNext++ 512×512 checkpoint is in `gs://gresearch/tapnextpp/`.
- Checkpoints are distributed via Google Cloud Storage and HuggingFace. The `NumpyFileCheckpointer` format is a dict with `params` and `state` keys saved via `np.save`.
- **JAX→PyTorch checkpoint conversion**: `tapnext/tapnext_torch_utils.py::restore_model_from_jax_checkpoint` manually maps JAX `.npz` keys to PyTorch parameter names (Haiku `.`→`/` separator conversion, Conv kernel permute `(3,2,0,1)`, norm scale/bias remapping).
- `ParameterizedTAPIR` (in `tapir_model.py`) wraps the Haiku module into a callable class — it monkey-patches methods to inject Haiku params/state automatically. This is the standard way to use TAPIR for inference.
- For multi-resolution evaluation, provide `refinement_resolutions` (list of (H,W) tuples). The model runs PIPs refinement at each resolution; final output is averaged across the last iteration of each resolution level.
- TAPVid-3D has a separate license from the rest of the repo (see `tapnet/tapvid3d/LICENSE`).
