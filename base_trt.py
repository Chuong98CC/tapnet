from __future__ import annotations

import numpy as np
import cv2
import tensorrt as trt
import torch
from abc import ABC


# Convert numpy dtype name to torch dtype
def trt_to_torch_dtype(trt_dtype):
    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.UINT8: torch.uint8,
    }
    return dtype_map.get(trt_dtype, torch.float32)


class TRTModel(ABC):
    """TensorRT engine wrapper.

    Owns engine load, IO-tensor metadata, input-geometry resolution (4-D NCHW
    for mono/stereo, 5-D (1,N,3,H,W) for any-view), and a single name-matched
    execution helper ``_run``.  Kept image-count-agnostic so monocular, stereo,
    and any-view models all share it.  DA3-specific pre/post lives in
    ``BaseDA3Model``.
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.context = None
        self.engine = None

        print(f"Loading engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {engine_path}.\n"
                f"The engine is incompatible with TensorRT {trt.__version__}.\n"
                f"Rebuild the engine with your current TensorRT version."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Store tensor info for TensorRT 10.3+ (no manual memory allocation).
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_info = {"name": tensor_name, "shape": shape, "dtype": dtype}
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
                print(f"Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
            else:
                self.outputs.append(tensor_info)
                print(f"Output: {tensor_name}, shape: {shape}, dtype: {dtype}")

        self._resolve_input_geometry()

    def _resolve_input_geometry(self) -> None:
        """Resolve target H/W (and num_views) from the first input shape.

        Supports 5-D any-view ``(1, N, 3, H, W)`` and 4-D ``(1, 3, H, W)``.
        Dynamic dims (TRT reports ``-1``) are treated as unresolved.
        """
        shape = self.inputs[0]["shape"]

        def _dim(idx: int) -> int | None:
            if idx >= len(shape):
                return None
            v = int(shape[idx])
            return v if v > 0 else None

        if len(shape) == 5:  # (B, N, 3, H, W) — any-view
            self.num_views = _dim(1)
            self.target_h = _dim(3)
            self.target_w = _dim(4)
        elif len(shape) == 4:  # (B, 3, H, W) — mono / stereo / metric
            self.num_views = None
            self.target_h = _dim(2)
            self.target_w = _dim(3)
        else:
            raise ValueError(
                f"Unexpected input rank {len(shape)} for " f"{self.inputs[0]['name']}: {shape}"
            )

        if self.target_h is None or self.target_w is None:
            raise ValueError(
                f"Engine input '{self.inputs[0]['name']}' has non-static H/W "
                f"{shape}; rebuild with fixed height/width."
            )

    @property
    def input_width(self):
        return self.target_w

    @property
    def input_height(self):
        return self.target_h

    def resize_img(self, img: np.ndarray):
        """General single-image letterbox (aspect-preserve + center-pad).

        Returns a padded uint8 image and meta.  Used by the non-DA3 mono/stereo
        paths; the DA3 classes use ``BaseDA3Model.preprocess_views`` instead.
        """
        orig_h, orig_w = img.shape[:2]
        scale_w = self.target_w / orig_w
        scale_h = self.target_h / orig_h
        raw_scale = min(scale_w, scale_h)
        scale_factor = np.floor(raw_scale * 100.0) / 100.0
        if scale_factor <= 0:
            scale_factor = raw_scale

        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale_factor),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return img_padded, meta

    def img2tensor(self, img: np.ndarray):
        """HWC image → (1, 3, H, W) CUDA tensor at the first input's dtype."""
        input_dtype = trt_to_torch_dtype(self.inputs[0]["dtype"])
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        return tensor.cuda().to(input_dtype)

    def _to_input_tensor(self, array, trt_dtype) -> torch.Tensor:
        """Move an already model-shaped numpy/torch array to a contiguous CUDA
        tensor of the engine input's dtype (no layout change)."""
        if isinstance(array, torch.Tensor):
            t = array
        else:
            t = torch.from_numpy(np.ascontiguousarray(array))
        return t.cuda().to(trt_to_torch_dtype(trt_dtype)).contiguous()

    def _run(self, named_inputs: dict, np_output: bool = True) -> dict:
        """Generic execution: bind inputs BY NAME, allocate outputs, execute.

        ``named_inputs`` maps each engine input name to an array already in the
        engine's expected layout (e.g. ``image`` (1,N,3,H,W), ``extrinsics``
        (1,N,4,4)).  Returns ``{output_name: array}`` (numpy if ``np_output``).
        """
        input_tensors = []  # keep alive until execution completes
        for inp in self.inputs:
            name = inp["name"]
            if name not in named_inputs:
                raise KeyError(f"Missing engine input '{name}' in named_inputs")
            t = self._to_input_tensor(named_inputs[name], inp["dtype"])
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors.append(t)

        output_tensors = {}
        for out in self.outputs:
            name = out["name"]
            shape = self.context.get_tensor_shape(name)
            dtype = trt_to_torch_dtype(out["dtype"])
            t = torch.empty(tuple(shape), dtype=dtype, device="cuda").contiguous()
            output_tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference execution failed")
        torch.cuda.synchronize()

        return self.output2numpy(output_tensors) if np_output else output_tensors

    def output2numpy(self, output_tensors: dict):
        return {name: t.cpu().numpy() for name, t in output_tensors.items()}

    def __del__(self):
        if hasattr(self, "context") and self.context is not None:
            del self.context
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine

    def parse_outputs(self, *args, **kwargs):
        """Extract/crop/mask outputs for a frame. Overridden by subclasses."""
        pass


class MonoDepthTRT(TRTModel):
    def _infer(self, img: np.ndarray, np_output: bool = True):
        """Single-image inference. Input: uint8 HWC RGB."""
        img_tensor = self.img2tensor(img)
        return self._run({self.inputs[0]["name"]: img_tensor}, np_output)


class StereoDepthTRT(TRTModel):
    def _infer(self, left_img: np.ndarray, right_img: np.ndarray, np_output: bool = True):
        """Stereo inference. Two-input engine or concatenated 6-channel input."""
        left_tensor = self.img2tensor(left_img)
        right_tensor = self.img2tensor(right_img)
        if len(self.inputs) == 2:
            named = {
                self.inputs[0]["name"]: left_tensor,
                self.inputs[1]["name"]: right_tensor,
            }
        else:
            named = {self.inputs[0]["name"]: torch.cat([left_tensor, right_tensor], dim=1)}
        return self._run(named, np_output)

    def preprocess(self, left_img, right_img):
        left_resized, meta_info = self.resize_img(left_img)
        right_resized, _ = self.resize_img(right_img)
        return left_resized, right_resized, meta_info
