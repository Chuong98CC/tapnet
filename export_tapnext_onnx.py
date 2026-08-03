
"""Export TAPNext (the core of TAPNext++) to ONNX format.

The exported ONNX model processes a **single frame** and produces tracking
outputs plus updated recurrent state for the next frame. This is the core
inference step — the caller is responsible for the frame loop.

Usage::

    python export_tapnext_onnx.py \\
        --checkpoint checkpoints/tapnextpp_512.ckpt \\
        --output tapnext_512.onnx

Architecture notes
------------------
TAPNext is a recurrent model with 12 TRecViT blocks, each containing:

* An RG-LRU (Real-Gated Linear Recurrent Unit) with a float32 hidden state of
  shape ``[B*N, lru_width]``.
* A causal 1D convolution with a ring-buffer of 3 prior frames of shape
  ``[B*N, 3, lru_width]``.

Together these form the ``RecurrentBlockCache``. The full recurrent state
across all 12 blocks plus a step counter and the original query points is
wrapped in ``TAPNextTrackingState``.

Because ONNX cannot represent Python objects, this script **flattens** the
recurrent state into individual tensors:

* ``step`` — scalar int64, current frame index.
* ``query_points`` — ``[B, Q, 3]`` float32 ``[t, y, x]`` in model (256) space
  with original query timesteps.
* For each of the 12 blocks:
  - ``rg_lru_state_{i}`` — ``[B*N, 768]`` float32
  - ``conv1d_state_{i}``  — ``[B*N, 3, 768]`` float32

where ``N = num_patches + Q`` and ``num_patches = (input_resolution // 8)²``.

The exported model takes ``3 + 24 = 27`` inputs and returns
``4 + 24 = 28`` outputs (tracks, visible_logits, step_out, query_points_out,
plus 24 updated cache tensors).

Pre / post processing
---------------------
The exported model expects **preprocessed** float32 video in [-1, 1] range
and outputs tracks in **model (256×256) space** in [y, x] order.  Use the
helpers in ``tapnet.tapnextpp.votsp2026.utils`` or the reference code below
to convert between display space and model space.

.. code-block:: python

    # --- Preprocessing (BGR uint8 → model input) ---
    def preprocess_frame(frame_bgr, input_resolution=512):
        '''[H,W,3] uint8 BGR → [1,1,S,S,3] float32 in [-1,1].'''
        import torch.nn.functional as F
        frame_rgb = frame_bgr[..., ::-1].copy()
        t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(input_resolution, input_resolution),
                          mode='bilinear', align_corners=False)
        t = t.div_(127.5).sub_(1.0)
        return t.permute(0, 2, 3, 1).unsqueeze(0)  # [1, 1, S, S, 3]

    # --- Coordinate conversion ---
    def display_to_model(pts_xy, disp_h, disp_w, model_size=256):
        scale = np.array([model_size / disp_w, model_size / disp_h],
                         dtype=np.float32)
        return pts_xy * scale

    def model_to_display(pts_xy, disp_h, disp_w, model_size=256):
        scale = np.array([disp_w / model_size, disp_h / model_size],
                         dtype=np.float32)
        return pts_xy * scale

    # --- Building a query tensor ---
    def make_query_tensor(model_pts_xy, device='cpu'):
        '''model_pts_xy: [Q,2] [x,y] in model (256) space.'''
        q = len(model_pts_xy)
        query = np.zeros((q, 3), dtype=np.float32)
        query[:, 0] = 0.0       # timestep 0 = first frame
        query[:, 1] = model_pts_xy[:, 1]  # y
        query[:, 2] = model_pts_xy[:, 0]  # x
        return torch.from_numpy(query).unsqueeze(0)  # [1, Q, 3]
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from tapnet.tapnext import tapnext_lru_modules
from tapnet.tapnext import tapnext_torch
from tapnet.tapnext.tapnext_lru_modules import RecurrentBlockCache


# ---------------------------------------------------------------------------
# ONNX-exportable wrapper
# ---------------------------------------------------------------------------


class TAPNextONNXWrapper(nn.Module):
  """Wraps ``TAPNext.forward`` so all recurrent state is explicit tensors.

  The wrapper:

  1. Accepts flat SSM cache tensors (24 tensors for 12 blocks).
  2. Reconstructs a ``TAPNextTrackingState``.
  3. Calls the inner model with ``query_points=None`` (the state already
     carries the original query points).
  4. Flattens the updated state back into individual tensors.

  This makes the full computation traceable by ``torch.onnx.export``.
  """

  def __init__(self, model: tapnext_torch.TAPNext, num_blocks: int = 12):
    super().__init__()
    self.model = model
    self.num_blocks = num_blocks

  # fmt: off
  def forward(
      self,
      video: torch.Tensor,          # [B, 1, H, W, 3]  float32, preprocessed video frame
      query_points: torch.Tensor,   # [B, Q, 3]        float32, [t, y, x] in model-256 space
      step: torch.Tensor,           # scalar            int64, current frame index
      *cache_tensors: torch.Tensor, # 24 tensors        (12 blocks × 2: rg_lru + conv1d)
  ) -> Tuple[torch.Tensor, ...]:
  # fmt: on
    """Single-frame forward pass.

    Returns:
        ``(tracks, visible_logits, step_out, query_points_out, *cache_out)``

        * **tracks** ``[B, 1, Q, 2]`` — float32 ``[y, x]`` in model (256) space.
        * **visible_logits** ``[B, 1, Q, 1]`` — float32, >0 ⇒ visible.
        * **step_out** — scalar int64, ``step + 1``.
        * **query_points_out** ``[B, Q, 3]`` — unchanged query points
          (same as input, carried forward for the next step).
        * **cache_out** — 24 tensors, updated SSM caches.
    """
    # -- Reconstruct hidden state from flat tensor list -----------------
    hidden_state: List[RecurrentBlockCache] = []
    for i in range(self.num_blocks):
      hidden_state.append(
          RecurrentBlockCache(
              rg_lru_state=cache_tensors[2 * i],
              conv1d_state=cache_tensors[2 * i + 1],
          )
      )

    # step may be a 0-d int64 tensor; the model uses it in arithmetic
    # (subtraction from query timesteps and increment by 1), so keeping
    # it as a tensor is ONNX-friendly.
    state = tapnext_torch.TAPNextTrackingState(
        step=step,
        query_points=query_points,
        hidden_state=hidden_state,
    )

    # -- Call the inner model -------------------------------------------
    # When ``state`` is provided the model ignores its ``query_points``
    # argument and uses ``state.query_points`` instead (after adjusting
    # timesteps by ``-state.step``).  Passing ``None`` is safe.
    tracks, _track_logits, vis_logits, new_state = self.model(
        video=video,
        query_points=None,
        state=state,
    )

    # -- Flatten outputs ------------------------------------------------
    # Build output tuple in a fixed order so ONNX can assign stable names.
    outputs: List[torch.Tensor] = [
        tracks,              # [B, 1, Q, 2]
        vis_logits,          # [B, 1, Q, 1]
    ]

    # new_state.step is ``step + 1``.  Keep it as a scalar tensor.
    step_out = new_state.step
    # Under the classic ONNX tracer the shape may be lost — ensure 0-d.
    if isinstance(step_out, torch.Tensor) and step_out.ndim > 0:
      step_out = step_out.reshape(())
    outputs.append(step_out)

    outputs.append(new_state.query_points)   # [B, Q, 3]

    for cache in new_state.hidden_state:
      outputs.append(cache.rg_lru_state)     # [B*N, 768]
      outputs.append(cache.conv1d_state)     # [B*N, 3, 768]

    return tuple(outputs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_inner_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tapnext_torch.TAPNext:
  """Load the TAPNext module from a TAPNext++ checkpoint.

  Handles Lightning-style ``"tapnext."`` key prefixes that may be present
  in training checkpoints.
  """
  # MODEL_SIZE is always 256 — coordinate heads are trained at this resolution.
  inner = tapnext_torch.TAPNext(image_size=(256, 256))

  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  state_dict = ckpt.get("state_dict", ckpt)
  # Strip Lightning "tapnext." prefix if present.
  state_dict = {k.removeprefix("tapnext."): v for k, v in state_dict.items()}
  inner.load_state_dict(state_dict)

  inner = inner.to(device)
  inner.eval()
  return inner


# ---------------------------------------------------------------------------
# Dummy inputs for tracing
# ---------------------------------------------------------------------------

def make_dummy_inputs(
    model: tapnext_torch.TAPNext,
    input_resolution: int = 512,
    num_queries: int = 65,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
  """Build example inputs that exercise the full recurrent path.

  Args:
      model: The loaded TAPNext model.
      input_resolution: Spatial size of the preprocessed video frame.
      num_queries: Number of query points (real + support).  65 = 1 real +
          64 support as used by the VOTSp2026 tracker.
      device: Target device.

  Returns:
      ``(video, query_points, step, cache_tensors)`` suitable for tracing.
  """
  b = 1
  t = 1  # single frame
  h = w = input_resolution
  q = num_queries

  patch_size = model.patch_size[0]  # 8
  patches_h = h // patch_size
  patches_w = w // patch_size
  num_patches = patches_h * patches_w
  total_tokens = num_patches + q  # N in the model

  lru_width = model.width  # 768

  # Preprocessed video frame in [-1, 1] range.
  video = torch.randn(b, t, h, w, 3, device=device, dtype=torch.float32)

  # Query points in model-256 space: [t=0, y=center, x=center].
  query_points = torch.zeros(b, q, 3, device=device, dtype=torch.float32)
  query_points[..., 0] = 0.0       # timestep (frame 0)
  query_points[..., 1] = 256 / 2   # y
  query_points[..., 2] = 256 / 2   # x

  # Use step > 0 so the trace exercises the full "past query" code path
  # (query timesteps shifted negative, mask tokens used instead of point
  # query tokens).
  step = torch.tensor(1, dtype=torch.int64, device=device)

  # Initial SSM caches — all zeros (as if starting from the first frame).
  cache_tensors: List[torch.Tensor] = []
  for _ in range(12):
    # RG-LRU hidden state: float32, shape [B*N, lru_width].
    cache_tensors.append(
        torch.zeros(b * total_tokens, lru_width,
                    device=device, dtype=torch.float32)
    )
    # Causal conv ring-buffer: shape [B*N, 3, lru_width].
    cache_tensors.append(
        torch.zeros(b * total_tokens, 3, lru_width,
                    device=device, dtype=torch.float32)
    )

  return video, query_points, step, cache_tensors


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _build_io_names(num_blocks: int = 12):
  """Return ``(input_names, output_names)`` for the ONNX graph."""
  input_names = ["video", "query_points", "step"]
  for i in range(num_blocks):
    input_names.append(f"rg_lru_state_{i}")
    input_names.append(f"conv1d_state_{i}")

  output_names = ["tracks", "visible_logits", "step_out", "query_points_out"]
  for i in range(num_blocks):
    output_names.append(f"rg_lru_state_out_{i}")
    output_names.append(f"conv1d_state_out_{i}")

  return input_names, output_names


def _build_dynamic_axes(
    input_resolution: int,
    patch_size: int = 8,
    num_blocks: int = 12,
) -> dict:
  """Return dynamic-axis specifications for variable batch / queries.

  Used only by the classic (``dynamo=False``) export path.
  """
  dynamic_axes: dict = {
      "video":            {0: "batch"},
      "query_points":     {0: "batch", 1: "num_queries"},
      "tracks":           {0: "batch", 2: "num_queries"},
      "visible_logits":   {0: "batch", 2: "num_queries"},
      "query_points_out": {0: "batch", 1: "num_queries"},
  }

  for i in range(num_blocks):
    dynamic_axes[f"rg_lru_state_{i}"]     = {0: "batch_times_tokens"}
    dynamic_axes[f"conv1d_state_{i}"]      = {0: "batch_times_tokens"}
    dynamic_axes[f"rg_lru_state_out_{i}"]  = {0: "batch_times_tokens"}
    dynamic_axes[f"conv1d_state_out_{i}"]  = {0: "batch_times_tokens"}

  return dynamic_axes


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    input_resolution: int = 512,
    num_queries: int = 65,
    opset_version: int = 17,
    verify: bool = True,
) -> None:
  """Export the TAPNext model to ONNX.

  Uses the classic TorchScript-based ONNX exporter.  The exported model
  **requires** the same ``num_queries`` at runtime as at export time
  because ViT attention ``Reshape`` nodes bake in the total token count
  ``N = num_patches + num_queries``.

  For variable query counts, pad queries to a fixed count (e.g. 65 for
  the VOT tracker pattern of 1 real + 64 support points) and ignore the
  extra track outputs.

  Args:
      checkpoint_path: Path to ``.ckpt`` or ``.pt`` checkpoint.
      output_path: Destination path for the ONNX file.
      input_resolution: Spatial size of preprocessed frames (256 or 512).
      num_queries: Number of query points for the dummy trace input.
      opset_version: ONNX opset version (17+ recommended for bicubic
          interpolation support).
      verify: If True, load the exported model and run ``onnx.checker``.
  """
  device = "cpu"
  num_blocks = 12

  # ------------------------------------------------------------------
  # 1. Load model
  # ------------------------------------------------------------------
  print(f"Loading checkpoint: {checkpoint_path}")
  model = load_inner_model(checkpoint_path, device=device)
  print(f"  Model loaded ({model.width=}, patch_size={model.patch_size})")

  # ------------------------------------------------------------------
  # 2. Create wrapper and dummy inputs
  # ------------------------------------------------------------------
  wrapper = TAPNextONNXWrapper(model, num_blocks=num_blocks)
  wrapper = wrapper.to(device)
  wrapper.eval()

  video, query_points, step, cache_tensors = make_dummy_inputs(
      model,
      input_resolution=input_resolution,
      num_queries=num_queries,
      device=device,
  )

  patches = (input_resolution // model.patch_size[0]) ** 2
  print(f"  Dummy inputs: resolution={input_resolution}, queries={num_queries}")
  print(f"  Patches={patches}, total_tokens={patches + num_queries}")

  # Build a flat tuple of all inputs (27 tensors).
  args = (video, query_points, step) + tuple(cache_tensors)

  # ------------------------------------------------------------------
  # 3. Build I/O metadata
  # ------------------------------------------------------------------
  input_names, output_names = _build_io_names(num_blocks)

  print(f"  Inputs:  {len(input_names)} tensors")
  print(f"  Outputs: {len(output_names)} tensors")

  # ------------------------------------------------------------------
  # 4. Export (classic TorchScript tracer)
  # ------------------------------------------------------------------
  print(f"Exporting to {output_path}  (opset={opset_version}) ...")

  dynamic_axes = _build_dynamic_axes(input_resolution,
                                     model.patch_size[0], num_blocks)
  torch.onnx.export(
      wrapper,
      args,
      output_path,
      input_names=input_names,
      output_names=output_names,
      dynamic_axes=dynamic_axes,
      opset_version=opset_version,
      do_constant_folding=True,
      export_params=True,
      verbose=False,
      dynamo=False,
  )
  print("  Export complete.")
  print(f"  NOTE: The exported model requires ``num_queries={num_queries}`` "
        "at runtime. For variable query counts, pad queries to this fixed "
        "count and ignore the extra outputs.")

  # ------------------------------------------------------------------
  # 5. Verify
  # ------------------------------------------------------------------
  if verify:
    print("Verifying ONNX model ...")
    try:
      import onnx
    except ImportError:
      print("  onnx not installed — skipping verification.", file=sys.stderr)
      print("  Install with: pip install onnx", file=sys.stderr)
      return

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    print("  Verification passed.")

    # Print a brief summary.
    print(f"\nONNX model summary:")
    print(f"  IR version:     {onnx_model.ir_version}")
    print(f"  Opset:          {onnx_model.opset_import[0].domain} "
          f"v{onnx_model.opset_import[0].version}")
    print(f"  Producer:       {onnx_model.producer_name}")
    graph = onnx_model.graph
    print(f"  Nodes:          {len(graph.node)}")
    print(f"  Initializers:   {len(graph.initializer)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(
      description="Export TAPNext (TAPNext++) to ONNX for single-frame inference."
  )
  parser.add_argument(
      "--checkpoint",
      default="checkpoints/tapnextpp_512.ckpt",
      help="Path to .ckpt / .pt checkpoint "
           "(default: checkpoints/tapnextpp_512.ckpt).",
  )
  parser.add_argument(
      "--output",
      default="tapnext_512.onnx",
      help="Output ONNX file path (default: tapnext_512.onnx).",
  )
  parser.add_argument(
      "--input-resolution",
      type=int,
      default=512,
      help="Spatial resolution of preprocessed frames (256 or 512). "
           "Default: 512 (for the tapnextpp_512 checkpoint).",
  )
  parser.add_argument(
      "--num-queries",
      type=int,
      default=65,
      help="Number of query points for the dummy trace input "
           "(default: 65 = 1 real + 64 support, matching the VOT tracker). "
           "The exported model requires exactly this many queries at runtime "
           "due to baked-in attention shapes.",
  )
  parser.add_argument(
      "--opset",
      type=int,
      default=17,
      help="ONNX opset version (default: 17).",
  )
  parser.add_argument(
      "--no-verify",
      action="store_true",
      help="Skip ONNX model verification.",
  )
  args = parser.parse_args()

  export_onnx(
      checkpoint_path=args.checkpoint,
      output_path=args.output,
      input_resolution=args.input_resolution,
      num_queries=args.num_queries,
      opset_version=args.opset,
      verify=not args.no_verify,
  )


if __name__ == "__main__":
  main()
