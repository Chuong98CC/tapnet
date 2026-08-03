"""Run TAPNext ONNX point tracking on a video and save a visualization.

Loads :class:`TapNextOnnx` from ``base_tapnext.py``, tracks a set of query
points across an input video frame-by-frame, and writes an MP4 with the
tracked points painted on using the same visualization as the TAP-Vid demo
(``tapnet.utils.viz_utils.paint_point_track``).

Query points for the first frame can be supplied three ways (checked in
priority order):

1. ``--points "x1,y1 x2,y2 ..."`` — explicit points in display pixels.
2. ``--bbox x0 y0 x1 y1 --grid N`` — an ``N x N`` uniform grid inside the box.
3. Neither — a default ``8 x 8`` grid over the whole first frame.

The exported ONNX model tracks a fixed number of query points
(``--num-queries``, 65 for ``tapnext_512.onnx``).  Fewer real points are
padded up to that count with dummy points that are tracked but not drawn.

Run with the conda ``tapnet`` environment (has onnxruntime / cv2 / mediapy)::

    /home/chuong/miniconda3/envs/tapnet/bin/python run_tapnext_inference.py \
        --video data/astribot_stereo_lrb/videos/observation.images.cam_head_stereo_left/chunk-000/file-000.mp4 \
        --bbox 200 150 600 500 --grid 8
"""

from __future__ import annotations

import argparse
import os
from typing import List

import mediapy as media
import numpy as np
import onnxruntime as ort

from base_tapnext import TapNextOnnx
from tapnet.utils import viz_utils

DEFAULT_VIDEO = (
    "data/astribot_stereo_lrb/videos/"
    "observation.images.cam_head_stereo_left/chunk-000/file-000.mp4"
)


# ---------------------------------------------------------------------------
# Query point construction
# ---------------------------------------------------------------------------

def parse_points(spec: str) -> np.ndarray:
    """Parse ``"x1,y1 x2,y2 ..."`` into an ``[N, 2]`` float32 array."""
    pts: List[List[float]] = []
    for pair in spec.split():
        try:
            x_str, y_str = pair.split(",")
            pts.append([float(x_str), float(y_str)])
        except ValueError as exc:
            raise ValueError(
                f"Could not parse point '{pair}'. Expected 'x,y' pairs "
                "separated by spaces, e.g. \"100,120 300,200\"."
            ) from exc
    if not pts:
        raise ValueError("--points was empty.")
    return np.asarray(pts, dtype=np.float32)


def grid_in_bbox(bbox: List[float], grid: int) -> np.ndarray:
    """Uniformly sample a ``grid x grid`` set of points inside ``bbox``.

    Points are inset from the edges (``np.linspace`` cell centers) so none
    land exactly on the box border.

    Parameters
    ----------
    bbox : list of float
        ``[x0, y0, x1, y1]`` in display pixels.
    grid : int
        Number of points per axis (total ``grid * grid``).

    Returns
    -------
    np.ndarray
        ``[grid * grid, 2]`` float32 ``[x, y]`` in display pixels.
    """
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    # Cell-center sampling: split each axis into `grid` cells, take centers.
    xs = x0 + (np.arange(grid) + 0.5) * (x1 - x0) / grid
    ys = y0 + (np.arange(grid) + 0.5) * (y1 - y0) / grid
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def build_query_points(
    args: argparse.Namespace, disp_w: int, disp_h: int
) -> np.ndarray:
    """Resolve query points (display px) from CLI args."""
    if args.points is not None:
        pts = parse_points(args.points)
        source = f"--points ({len(pts)} points)"
    elif args.bbox is not None:
        pts = grid_in_bbox(args.bbox, args.grid)
        source = f"--bbox {args.bbox} with {args.grid}x{args.grid} grid"
    else:
        # Default: grid over the whole frame, inset from the borders.
        pts = grid_in_bbox([0.0, 0.0, float(disp_w), float(disp_h)], args.grid)
        source = f"default {args.grid}x{args.grid} grid over full frame"

    print(f"[query] {len(pts)} query points from {source}")
    return pts


def pad_queries(pts: np.ndarray, num_queries: int) -> np.ndarray:
    """Pad ``[R, 2]`` real points up to exactly ``num_queries`` points.

    Dummy points are placed at (0, 0); they are tracked by the model but the
    caller only visualizes the first ``R``.
    """
    r = len(pts)
    if r > num_queries:
        raise ValueError(
            f"Got {r} query points but the ONNX model tracks at most "
            f"{num_queries}. Reduce the number of points / grid size."
        )
    if r == num_queries:
        return pts.astype(np.float32)
    pad = np.zeros((num_queries - r, 2), dtype=np.float32)
    return np.concatenate([pts.astype(np.float32), pad], axis=0)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def track_video(
    model: TapNextOnnx,
    frames_bgr: np.ndarray,
    query_points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run online tracking over all frames.

    Parameters
    ----------
    model : TapNextOnnx
        Loaded ONNX model.
    frames_bgr : np.ndarray
        ``[F, H, W, 3]`` uint8 BGR frames.
    query_points_xy : np.ndarray
        ``[num_queries, 2]`` float32 ``[x, y]`` in display pixels (already
        padded to the model's fixed query count).

    Returns
    -------
    positions : np.ndarray
        ``[F, num_queries, 2]`` float32 ``[x, y]`` in display pixels.
    visible : np.ndarray
        ``[F, num_queries]`` bool.
    """
    num_frames = len(frames_bgr)
    positions = np.zeros((num_frames, len(query_points_xy), 2), dtype=np.float32)
    visible = np.zeros((num_frames, len(query_points_xy)), dtype=bool)

    state = None
    for t in range(num_frames):
        if t == 0:
            pos, vis, state = model.track_frame(
                frames_bgr[t], query_points_xy=query_points_xy
            )
        else:
            pos, vis, state = model.track_frame(frames_bgr[t], state=state)
        positions[t] = pos
        visible[t] = vis
        if (t + 1) % 25 == 0 or t == num_frames - 1:
            print(f"[track] frame {t + 1}/{num_frames}")

    return positions, visible


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAPNext ONNX point tracking + visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", default="tapnext_512.onnx", help="Path to the ONNX model.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path (default: <video>_tracked.mp4).",
    )
    parser.add_argument(
        "--points",
        default=None,
        help='Explicit query points in display px: "x1,y1 x2,y2 ...".',
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Bounding box (display px) to sample a uniform grid inside.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=8,
        help="Grid points per axis for --bbox (or the default full-frame grid).",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--ort-log-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="ONNX Runtime log severity (0=verbose ... 4=fatal). Default 3 "
        "(errors only) hides the benign step_out shape warning.",
    )
    parser.add_argument(
        "--input-resolution",
        type=int,
        default=512,
        help="Resolution the ONNX model was exported for.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=65,
        help="Fixed query count the ONNX model was exported with.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to process (inclusive, 0-based).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to process (exclusive). Default: end of video.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (frame skip interval).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of (strided) frames processed.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output fps (default: source video fps).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # Raise ORT's log level (default: errors only) to hide the benign
    # "step_out shape {} vs {1}" warning emitted every inference call.
    ort.set_default_logger_severity(args.ort_log_level)

    # -- Read video (RGB uint8) ---------------------------------------------
    print(f"[io] reading {args.video}")
    video = media.read_video(args.video)
    src_fps = getattr(getattr(video, "metadata", None), "fps", None)
    frames_rgb = np.asarray(video)
    total_read = len(frames_rgb)
    if total_read == 0:
        raise ValueError("Video contains no frames.")

    # Select frame range and stride: [start:end:stride], then optional cap.
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    end = args.end_frame if args.end_frame is not None else total_read
    frames_rgb = frames_rgb[args.start_frame : end : args.stride]
    if args.max_frames is not None:
        frames_rgb = frames_rgb[: args.max_frames]
    if len(frames_rgb) == 0:
        raise ValueError(
            f"No frames selected from {total_read} total "
            f"(start={args.start_frame}, end={args.end_frame}, stride={args.stride})."
        )
    frames_rgb = frames_rgb[..., :3].astype(np.uint8)
    num_frames, disp_h, disp_w = frames_rgb.shape[:3]
    print(
        f"[io] {num_frames} frames selected of {total_read} "
        f"(start={args.start_frame}, end={end}, stride={args.stride}) at {disp_w}x{disp_h}"
    )

    # track_frame expects BGR (it internally flips BGR->RGB).
    frames_bgr = frames_rgb[..., ::-1]

    # -- Build query points --------------------------------------------------
    real_pts = build_query_points(args, disp_w, disp_h)
    num_real = len(real_pts)
    padded = pad_queries(real_pts, args.num_queries)

    # -- Load model & track --------------------------------------------------
    if args.device == "cuda" and hasattr(ort, "preload_dlls"):
        # Load CUDA / cuDNN shared libs from the nvidia-*-cu12 pip packages so
        # the CUDA execution provider can be created.
        try:
            ort.preload_dlls()
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[warn] ort.preload_dlls() failed: {exc}")

    model = TapNextOnnx(
        args.onnx,
        device=args.device,
        input_resolution=args.input_resolution,
        num_queries=args.num_queries,
    )

    active = model.session.get_providers()
    print(f"[onnx] active providers: {active}")
    if args.device == "cuda" and "CUDAExecutionProvider" not in active:
        print(
            "[warn] --device cuda requested but the CUDA provider is NOT active; "
            "running on CPU. Install CUDA 12 + cuDNN 9 runtime libs into this "
            "environment (e.g. pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 "
            "nvidia-cufft-cu12 nvidia-curand-cu12)."
        )

    positions, visible = track_video(model, frames_bgr, padded)

    # Keep only the real (user-supplied) points for visualization.
    positions = positions[:, :num_real]   # [F, R, 2]
    visible = visible[:, :num_real]        # [F, R]

    # -- Visualize (paint_point_track wants [R, F, 2] / [R, F]) --------------
    point_tracks = np.transpose(positions, (1, 0, 2))   # [R, F, 2]
    visibles = np.transpose(visible, (1, 0))            # [R, F]
    print(f"[viz] painting {num_real} tracks over {num_frames} frames")
    painted = viz_utils.paint_point_track(frames_rgb, point_tracks, visibles)

    # -- Write ---------------------------------------------------------------
    output = args.output
    if output is None:
        base, _ = os.path.splitext(args.video)
        output = base + "_tracked.mp4"
    # Divide by stride so a subsampled video plays back at real-time speed.
    out_fps = args.fps if args.fps is not None else (src_fps or 25) / args.stride
    media.write_video(output, painted, fps=out_fps)
    print(f"[done] wrote {output} ({num_frames} frames, {num_real} points, {out_fps} fps)")


if __name__ == "__main__":
    main()
