"""ONNX Runtime base wrapper for Depth Anything 3 models.

Sibling of ``base_trt.TRTModel``: owns the ONNX Runtime session, exposes input/
output tensor metadata, resolves the model's fixed input geometry, and runs
inference.  DA3-specific pre/post-processing lives in ``base_da3.BaseDA3Model``;
concrete models multiple-inherit both.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

# ONNX Runtime type strings → numpy dtypes (for feed casting).
_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


class ONNXModel:
    """Thin ONNX Runtime session wrapper.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file (external ``.data`` weights resolved by ORT).
    device : str
        ``"cuda"`` (CUDA EP with CPU fallback) or ``"cpu"``.
    """

    def __init__(self, onnx_path: str, device: str = "cuda") -> None:
        if device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[ONNX] Loaded {onnx_path}")
        print(f"[ONNX] Provider: {self.session.get_providers()[0]}")

        self.inputs = [
            {"name": i.name, "shape": i.shape, "dtype": _ORT_TO_NP.get(i.type)}
            for i in self.session.get_inputs()
        ]
        self.outputs = [{"name": o.name, "shape": o.shape} for o in self.session.get_outputs()]
        for i in self.inputs:
            print(f"[ONNX] Input : {i['name']}  shape={i['shape']}")
        for o in self.outputs:
            print(f"[ONNX] Output: {o['name']}  shape={o['shape']}")

        self._resolve_input_geometry()

    def _resolve_input_geometry(self) -> None:
        """Set ``target_h``/``target_w``/``num_views`` from the first input shape.

        Supports 5-D any-view input ``(B, N, 3, H, W)`` and 4-D metric input
        ``(B, 3, H, W)``.  Symbolic (non-int) dims become ``None``.
        """
        shape = self.inputs[0]["shape"]

        def _dim(idx: int) -> int | None:
            return shape[idx] if idx < len(shape) and isinstance(shape[idx], int) else None

        if len(shape) == 5:      # (B, N, 3, H, W) — any-view
            self.num_views = _dim(1)
            self.target_h = _dim(3)
            self.target_w = _dim(4)
        elif len(shape) == 4:    # (B, 3, H, W) — metric / mono
            self.num_views = None
            self.target_h = _dim(2)
            self.target_w = _dim(3)
        else:
            raise ValueError(
                f"Unexpected input rank {len(shape)} for {self.inputs[0]['name']}: {shape}"
            )

        # Preprocessing needs concrete H/W; fail clearly rather than defaulting.
        if self.target_h is None or self.target_w is None:
            raise ValueError(
                f"Input '{self.inputs[0]['name']}' has non-static H/W {shape}; "
                "export the model with fixed height/width."
            )

    @property
    def input_width(self) -> int | None:
        return self.target_w

    @property
    def input_height(self) -> int | None:
        return self.target_h

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the session, returning ``{output_name: array}``.

        Each fed array is cast to its input's declared dtype (mirroring
        ``TRTModel._run``), so callers can pass a single dtype regardless of what
        the graph expects (e.g. uint8 stereo inputs vs float32 DA3 inputs).
        """
        cast = {i["name"]: i["dtype"] for i in self.inputs}
        feed = {
            name: np.ascontiguousarray(arr).astype(dt) if (dt := cast.get(name)) is not None else arr
            for name, arr in feed.items()
        }
        names = [o["name"] for o in self.outputs]
        return dict(zip(names, self.session.run(names, feed)))
