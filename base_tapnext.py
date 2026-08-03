"""Base classes for TAPNext ONNX inference with pre/post-processing.

``BaseTapNext`` owns the geometry, coordinate conventions, and frame
preparation.  ``TapNextOnnx`` inherits from both :class:`ONNXModel` (from
``base_onnx.py``) and ``BaseTapNext``, providing a single ``track_frame``
call that takes a BGR uint8 frame and returns display-space tracks.

Typical usage::

    model = TapNextOnnx("tapnext_512.onnx", device="cuda")
    positions, visible, state = model.track_frame(frame0, query_points_xy)
    for frame in frames[1:]:
        positions, visible, state = model.track_frame(frame, state=state)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from base_onnx import ONNXModel

# ---------------------------------------------------------------------------
# Constants matching the exported ONNX model
# ---------------------------------------------------------------------------
MODEL_SIZE = 256          # Coordinate space for point positions
PATCH_SIZE = 8            # TRecViT patch size
LRU_WIDTH = 768           # Width of RG-LRU hidden state
NUM_BLOCKS = 12           # Number of TRecViT transformer blocks


# ---------------------------------------------------------------------------
# BaseTapNext — geometry & pre/post-processing
# ---------------------------------------------------------------------------

class BaseTapNext:
    """Pre- and post-processing helpers for TAPNext-family models.

    Handles frame normalisation, coordinate transforms between display
    pixels and model (256×256) space, query-tensor construction, and
    recurrent-state initialisation.

    Parameters
    ----------
    input_resolution : int
        Spatial size that frames are resized to before inference
        (e.g. 512 for the ``tapnextpp_512`` checkpoint).
    num_queries : int
        Fixed number of query points the ONNX model was exported with.
        Must match the ``--num-queries`` value used during export.
    num_blocks : int
        Number of TRecViT blocks (12 for all released checkpoints).
    lru_width : int
        Width of the RG-LRU hidden state (768).
    model_size : int
        Coordinate space size (always 256 — baked into the checkpoint).
    """

    def __init__(
        self,
        input_resolution: int = 512,
        num_queries: int = 65,
        num_blocks: int = NUM_BLOCKS,
        lru_width: int = LRU_WIDTH,
        model_size: int = MODEL_SIZE,
    ) -> None:
        self.input_resolution = input_resolution
        self.num_queries = num_queries
        self.num_blocks = num_blocks
        self.lru_width = lru_width
        self.model_size = model_size

        # Number of visual patches for the given resolution.
        self.num_patches = (input_resolution // PATCH_SIZE) ** 2
        # Total tokens N = patches + queries — determines cache tensor shapes.
        self.total_tokens = self.num_patches + num_queries

    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR uint8 frame to a model-ready tensor.

        Steps: BGR→RGB, resize to ``input_resolution``, normalise to
        [-1, 1], then reshape to ``[1, 1, S, S, 3]`` (batch, time,
        height, width, channels).

        Parameters
        ----------
        frame_bgr : np.ndarray
            ``[H, W, 3]`` uint8 BGR image (OpenCV layout).

        Returns
        -------
        np.ndarray
            ``[1, 1, S, S, 3]`` float32 tensor in [-1, 1].
        """
        import cv2

        # BGR → RGB
        frame_rgb = frame_bgr[..., ::-1].copy()

        # Resize to input_resolution × input_resolution.
        if frame_rgb.shape[0] != self.input_resolution or frame_rgb.shape[1] != self.input_resolution:
            frame_rgb = cv2.resize(
                frame_rgb,
                (self.input_resolution, self.input_resolution),
                interpolation=cv2.INTER_LINEAR,
            )

        # Normalise: [0, 255] → [-1, 1].
        t = frame_rgb.astype(np.float32)
        t = (t / 127.5) - 1.0

        # [H, W, 3] → [1, 1, H, W, 3].
        return t[np.newaxis, np.newaxis, ...]

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def display_to_model(
        self, pts_xy: np.ndarray, disp_h: int, disp_w: int
    ) -> np.ndarray:
        """Scale ``[x, y]`` points from display pixels to model (256) space.

        Parameters
        ----------
        pts_xy : np.ndarray
            ``[N, 2]`` float32 ``[x, y]`` in display pixels.
        disp_h : int
            Display frame height.
        disp_w : int
            Display frame width.

        Returns
        -------
        np.ndarray
            ``[N, 2]`` float32 ``[x, y]`` in model (256) space.
        """
        scale = np.array(
            [self.model_size / disp_w, self.model_size / disp_h],
            dtype=np.float32,
        )
        return (pts_xy * scale).astype(np.float32)

    def model_to_display(
        self, pts_xy: np.ndarray, disp_h: int, disp_w: int
    ) -> np.ndarray:
        """Scale ``[x, y]`` points from model (256) space to display pixels.

        Parameters
        ----------
        pts_xy : np.ndarray
            ``[N, 2]`` float32 ``[x, y]`` in model (256) space.
        disp_h : int
            Display frame height.
        disp_w : int
            Display frame width.

        Returns
        -------
        np.ndarray
            ``[N, 2]`` float32 ``[x, y]`` in display pixels.
        """
        scale = np.array(
            [disp_w / self.model_size, disp_h / self.model_size],
            dtype=np.float32,
        )
        return (pts_xy * scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Query tensor construction
    # ------------------------------------------------------------------

    def make_query_tensor(
        self, model_pts_xy: np.ndarray, query_timestep: int = 0
    ) -> np.ndarray:
        """Build a ``[1, Q, 3]`` query tensor in ``[t, y, x]`` format.

        The query count *must* equal ``self.num_queries`` — pad with
        dummy points (e.g. at (0, 0) with a large negative timestep to
        mask them) if you have fewer real queries.

        Parameters
        ----------
        model_pts_xy : np.ndarray
            ``[Q, 2]`` float32 ``[x, y]`` in model (256) space.
        query_timestep : int
            Frame index for the query points (0 = first frame).

        Returns
        -------
        np.ndarray
            ``[1, Q, 3]`` float32 with layout ``[t, y, x]``.
        """
        q = len(model_pts_xy)
        if q != self.num_queries:
            raise ValueError(
                f"Expected {self.num_queries} query points, got {q}. "
                "Pad queries to the ONNX model's fixed query count."
            )

        query = np.zeros((q, 3), dtype=np.float32)
        query[:, 0] = query_timestep          # t
        query[:, 1] = model_pts_xy[:, 1]      # y
        query[:, 2] = model_pts_xy[:, 0]      # x
        return query[np.newaxis, ...]          # [1, Q, 3]

    # ------------------------------------------------------------------
    # Recurrent state management
    # ------------------------------------------------------------------

    def init_cache(self) -> Dict[str, np.ndarray]:
        """Return zero-initialised SSM cache tensors.

        Call this once before the first frame.

        Returns
        -------
        dict
            Keys ``rg_lru_state_{i}`` and ``conv1d_state_{i}`` for
            *i* in 0…11, each a zero-filled float32 array.
        """
        cache: Dict[str, np.ndarray] = {}
        for i in range(self.num_blocks):
            cache[f"rg_lru_state_{i}"] = np.zeros(
                (self.total_tokens, self.lru_width), dtype=np.float32
            )
            cache[f"conv1d_state_{i}"] = np.zeros(
                (self.total_tokens, 3, self.lru_width), dtype=np.float32
            )
        return cache

    def init_state(
        self,
        query_points_xy: np.ndarray,
        disp_h: int,
        disp_w: int,
    ) -> Dict[str, Any]:
        """Build the initial recurrent state for the first frame.

        Parameters
        ----------
        query_points_xy : np.ndarray
            ``[Q, 2]`` float32 ``[x, y]`` in display pixels.
        disp_h : int
            Display frame height.
        disp_w : int
            Display frame width.

        Returns
        -------
        dict
            State dict with keys ``"step"`` (int), ``"query_points"``
            (``[1, Q, 3]`` float32), and ``"cache"`` (dict of zero
            arrays).
        """
        model_pts = self.display_to_model(query_points_xy, disp_h, disp_w)
        return {
            "step": np.array(0, dtype=np.int64),
            "query_points": self.make_query_tensor(model_pts, query_timestep=0),
            "cache": self.init_cache(),
        }

    def update_state(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Extract the updated recurrent state from ONNX outputs.

        Parameters
        ----------
        outputs : dict
            Raw output dict from :meth:`ONNXModel.run`.

        Returns
        -------
        dict
            State dict suitable as the ``state`` argument to
            :meth:`TapNextOnnx.track_frame`.
        """
        cache: Dict[str, np.ndarray] = {}
        for i in range(self.num_blocks):
            cache[f"rg_lru_state_{i}"] = outputs[f"rg_lru_state_out_{i}"]
            cache[f"conv1d_state_{i}"] = outputs[f"conv1d_state_out_{i}"]

        # step_out may be returned as a 1-d array [N] by some ONNX
        # runtimes — squeeze to a true scalar for the next iteration.
        step_out = outputs["step_out"]
        if step_out.ndim > 0:
            step_out = step_out.reshape(())

        return {
            "step": step_out,
            "query_points": outputs["query_points_out"],
            "cache": cache,
        }

    # ------------------------------------------------------------------
    # Output postprocessing
    # ------------------------------------------------------------------

    def postprocess_tracks(
        self,
        outputs: Dict[str, np.ndarray],
        disp_h: int,
        disp_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ONNX outputs to display-space positions and visibility.

        Parameters
        ----------
        outputs : dict
            Raw output dict from :meth:`ONNXModel.run`.
        disp_h : int
            Display frame height.
        disp_w : int
            Display frame width.

        Returns
        -------
        positions_xy : np.ndarray
            ``[Q, 2]`` float32 ``[x, y]`` in display pixels.
        visible : np.ndarray
            ``[Q]`` bool — ``True`` where the model predicts the point
            is visible.
        """
        # tracks: [1, 1, Q, 2] in [y, x] model-256 space
        tracks_yx = outputs["tracks"][0, 0]                    # [Q, 2]
        tracks_xy = tracks_yx[:, ::-1].copy()                  # [y,x] → [x,y]

        positions_xy = self.model_to_display(tracks_xy, disp_h, disp_w)

        # visible_logits: [1, 1, Q, 1] — logit > 0 ⇒ visible
        visible = outputs["visible_logits"][0, 0, :, 0] > 0.0

        return positions_xy, visible


# ---------------------------------------------------------------------------
# TapNextOnnx — combined ONNX + TAPNext wrapper
# ---------------------------------------------------------------------------

class TapNextOnnx(ONNXModel, BaseTapNext):
    """TAPNext ONNX runtime with integrated pre/post-processing.

    Loads an ONNX model exported by ``export_tapnext_onnx.py`` and exposes
    a single :meth:`track_frame` call that mirrors the PyTorch
    ``TAPNextPP.track_frame`` interface.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file.
    device : str
        ``"cuda"`` (CUDA EP with CPU fallback) or ``"cpu"``.
    input_resolution : int
        Spatial size used during ONNX export (512 for
        ``tapnextpp_512.ckpt``, 256 for standard checkpoints).
    num_queries : int
        Fixed query count the ONNX model was exported with.
    """

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        input_resolution: int = 512,
        num_queries: int = 65,
    ) -> None:
        # Initialise BaseTapNext first so geometry attributes are available.
        BaseTapNext.__init__(
            self,
            input_resolution=input_resolution,
            num_queries=num_queries,
        )
        ONNXModel.__init__(self, onnx_path, device=device)

    # ------------------------------------------------------------------
    # Geometry resolution from the ONNX graph
    # ------------------------------------------------------------------

    def _resolve_input_geometry(self) -> None:
        """Override :meth:`ONNXModel._resolve_input_geometry`.

        The TAPNext ONNX model has a 5-D input ``[B, 1, H, W, 3]``.
        We extract H/W and validate against the configured
        ``input_resolution``.
        """
        shape = self.inputs[0]["shape"]  # e.g. ['batch', 1, 'height', 'width', 3]

        # Dynamic dims appear as strings; concrete dims as ints.
        h = shape[2] if isinstance(shape[2], int) else None
        w = shape[3] if isinstance(shape[3], int) else None

        if h is not None and h != self.input_resolution:
            print(
                f"[TapNextOnnx] WARNING: ONNX input H={h} differs from "
                f"configured input_resolution={self.input_resolution}. "
                "Using ONNX value."
            )
            self.input_resolution = h
        if w is not None and w != self.input_resolution:
            print(
                f"[TapNextOnnx] WARNING: ONNX input W={w} differs from "
                f"configured input_resolution={self.input_resolution}. "
                "Using ONNX value."
            )
            self.input_resolution = w

        if h is not None:
            self.target_h = h
        if w is not None:
            self.target_w = w

        # Recompute patch/token counts if resolution changed.
        self.num_patches = (self.input_resolution // PATCH_SIZE) ** 2
        self.total_tokens = self.num_patches + self.num_queries

    @property
    def input_width(self) -> int:
        return self.input_resolution

    @property
    def input_height(self) -> int:
        return self.input_resolution

    # ------------------------------------------------------------------
    # Main tracking interface
    # ------------------------------------------------------------------

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        query_points_xy: np.ndarray | None = None,
        state: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Process one frame and return display-space tracking results.

        Frame-by-frame online tracking loop::

            model = TapNextOnnx("tapnext_512.onnx")
            positions, visible, state = model.track_frame(
                frame0, query_points_xy=query_xy
            )
            for frame in video[1:]:
                positions, visible, state = model.track_frame(frame, state=state)

        Parameters
        ----------
        frame_bgr : np.ndarray
            ``[H, W, 3]`` uint8 BGR frame (OpenCV layout).
        query_points_xy : np.ndarray or None
            ``[Q, 2]`` float32 ``[x, y]`` in display pixels.  Required
            when ``state`` is ``None`` (first frame); ignored otherwise.
        state : dict or None
            Recurrent state returned by a previous call.  ``None`` on
            the first frame.

        Returns
        -------
        positions_xy : np.ndarray
            ``[Q, 2]`` float32 ``[x, y]`` in display pixels.
        visible : np.ndarray
            ``[Q]`` bool — ``True`` where the model predicts visibility.
        state : dict
            Updated recurrent state for the next frame.
        """
        if query_points_xy is None and state is None:
            raise ValueError(
                "Either query_points_xy or state must be provided."
            )

        h, w = frame_bgr.shape[:2]

        # -- Prepare inputs -------------------------------------------------
        if state is None:
            # First frame: initialise state from query points.
            state = self.init_state(query_points_xy, h, w)

        # Preprocess frame.
        video = self.preprocess_frame(frame_bgr)

        # Build feed dict (order-insensitive — ONNX matches by name).
        feed: Dict[str, np.ndarray] = {
            "video": video,
            "query_points": state["query_points"],
            "step": state["step"],
            **state["cache"],
        }

        # -- Run inference --------------------------------------------------
        outputs = self.run(feed)

        # -- Postprocess ----------------------------------------------------
        positions_xy, visible = self.postprocess_tracks(outputs, h, w)

        # -- Build next state -----------------------------------------------
        new_state = self.update_state(outputs)

        return positions_xy, visible, new_state


# ---------------------------------------------------------------------------
# Convenience: load from checkpoint via the export script path
# ---------------------------------------------------------------------------

def load_tapnext_onnx(
    onnx_path: str,
    device: str = "cuda",
    input_resolution: int = 512,
    num_queries: int = 65,
) -> TapNextOnnx:
    """Load a TAPNext ONNX model.

    Convenience wrapper around :class:`TapNextOnnx`.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file.
    device : str
        ``"cuda"`` or ``"cpu"``.
    input_resolution : int
        Spatial resolution the model was exported for.
    num_queries : int
        Fixed query count the model was exported with.

    Returns
    -------
    TapNextOnnx
        Ready-to-use model instance.
    """
    return TapNextOnnx(
        onnx_path=onnx_path,
        device=device,
        input_resolution=input_resolution,
        num_queries=num_queries,
    )
