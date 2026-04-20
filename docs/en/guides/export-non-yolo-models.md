---
comments: true
description: Export any PyTorch model (timm, torchvision, or custom) to ONNX, OpenVINO, CoreML, TensorFlow SavedModel, TorchScript, NCNN, MNN, PaddlePaddle, and ExecuTorch with Ultralytics export utilities.
keywords: export PyTorch model, convert PyTorch to ONNX, PyTorch to CoreML, PyTorch to OpenVINO, PyTorch to TensorFlow, non-YOLO export, timm export, torchvision export, TorchScript export, NCNN export, MNN export, PaddlePaddle export, ExecuTorch export, TFLite export, Ultralytics export utilities, torch.nn.Module export, model conversion, model deployment, pytorch deployment
---

# How to Export Non-YOLO PyTorch Models with Ultralytics

Deploying PyTorch models to production usually means juggling a different exporter for every target: `torch.onnx.export` for ONNX, `coremltools` for Apple devices, `onnx2tf` for TensorFlow, `pnnx` for NCNN, and so on. Each tool has its own API, dependency quirks, and output conventions.

Ultralytics ships standalone export utilities that wrap all of these behind one consistent interface. You can export any `torch.nn.Module`, including [timm](https://github.com/huggingface/pytorch-image-models) image models, [torchvision](https://pytorch.org/vision/) classifiers and detectors, or your own custom architectures, to [ONNX](../integrations/onnx.md), [TorchScript](../integrations/torchscript.md), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), [NCNN](../integrations/ncnn.md), [PaddlePaddle](../integrations/paddlepaddle.md), [MNN](../integrations/mnn.md), [ExecuTorch](../integrations/executorch.md), and [TensorFlow SavedModel](../integrations/tf-savedmodel.md) without learning each backend separately.

This guide walks through every supported format with working code, explains the dependencies and platform constraints for each, and shows how to verify your exported model matches the original PyTorch output.

## Why Use Ultralytics for Non-YOLO Export?

- **One API across 9+ formats:** learn a single calling convention instead of a dozen.
- **Automatic dependency resolution:** missing packages (`onnx`, `coremltools`, `openvino`, `MNN`, `pnnx`, etc.) are detected and installed on first use.
- **Battle-tested export pipeline:** the same code path ships in production for every Ultralytics YOLO export.
- **FP16 and INT8 quantization** built in for formats that support it (OpenVINO, CoreML, MNN, NCNN).
- **Works on CPU:** no GPU required for the export step itself, so you can run it locally on any laptop.

## Supported Export Formats

All functions accept a standard `torch.nn.Module` and an example input tensor. No YOLO-specific attributes are required.

| Format          | Function              | Install                                            | Output                         |
| --------------- | --------------------- | -------------------------------------------------- | ------------------------------ |
| ONNX            | `torch2onnx()`        | `pip install onnx`                                 | `.onnx` file                   |
| TorchScript     | `torch2torchscript()` | included with PyTorch                              | `.torchscript` file            |
| OpenVINO        | `torch2openvino()`    | `pip install openvino`                             | `_openvino_model/` directory   |
| CoreML          | `torch2coreml()`      | `pip install coremltools`                          | `.mlpackage`                   |
| TF SavedModel   | `onnx2saved_model()`  | `pip install onnx2tf tensorflow tf_keras sng4onnx` | `_saved_model/` directory      |
| TF Frozen Graph | `keras2pb()`          | same as TF SavedModel                              | `.pb` file                     |
| NCNN            | `torch2ncnn()`        | `pip install ncnn pnnx`                            | `_ncnn_model/` directory       |
| MNN             | `onnx2mnn()`          | `pip install MNN`                                  | `.mnn` file                    |
| PaddlePaddle    | `torch2paddle()`      | `pip install paddlepaddle x2paddle`                | `_paddle_model/` directory     |
| ExecuTorch      | `torch2executorch()`  | `pip install executorch`                           | `_executorch_model/` directory |

!!! note "ONNX as an intermediate format"

    [MNN](../integrations/mnn.md), [TF SavedModel](../integrations/tf-savedmodel.md), and TF Frozen Graph exports go through ONNX as an intermediate step. Export to ONNX first, then convert.

## Quick Start

The fastest path is a two-line export to [ONNX](../integrations/onnx.md) with no YOLO code and no setup beyond `pip install ultralytics onnx`:

```python
import torch
import timm
from ultralytics.utils.export import torch2onnx

model = timm.create_model("resnet18", pretrained=True).eval()
torch2onnx(model, torch.randn(1, 3, 224, 224), output_file="resnet18.onnx")
```

For other formats, swap `torch2onnx` for the target function in the [format table](#supported-export-formats) above and adjust arguments. See the [step-by-step examples](#step-by-step-examples) below for each format.

## Step-by-Step Examples

Every example below uses the same setup, a pretrained ResNet-18 from timm in evaluation mode:

```python
import torch
import timm

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
```

!!! warning "Always call `model.eval()` before exporting"

    Dropout, [batch normalization](https://www.ultralytics.com/glossary/batch-normalization), and other train-only layers behave differently during inference. Skipping `.eval()` produces exports with incorrect outputs.

### Export to ONNX

```python
from ultralytics.utils.export import torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
```

For dynamic batch size, pass a `dynamic` dictionary:

```python
torch2onnx(model, im, output_file="resnet18_dyn.onnx", dynamic={"images": {0: "batch_size"}})
```

The default opset is `14` and the default input name is `"images"`. Override with the `opset`, `input_names`, or `output_names` arguments.

### Export to TorchScript

No extra dependencies needed. Uses `torch.jit.trace` under the hood.

```python
from ultralytics.utils.export import torch2torchscript

torch2torchscript(model, im, output_file="resnet18.torchscript")
```

### Export to OpenVINO

```python
from ultralytics.utils.export import torch2openvino

ov_model = torch2openvino(model, im, output_dir="resnet18_openvino_model")
```

The directory contains a fixed-name `model.xml` and `model.bin` pair. OpenVINO names the inputs after your model's `forward` argument names (typically `x` for generic models). Supports `half=True` for FP16 and `int8=True` for INT8 quantization (INT8 also requires a `calibration_dataset`). Requires `openvino>=2024.0.0` (or `>=2025.2.0` on macOS 15.4+) and `torch>=2.1`.

### Export to CoreML

```python
import coremltools as ct
from ultralytics.utils.export import torch2coreml

inputs = [ct.TensorType("input", shape=(1, 3, 224, 224))]
ct_model = torch2coreml(model, inputs, im, classifier_names=None, output_file="resnet18.mlpackage")
```

For [classification](https://www.ultralytics.com/glossary/image-classification) models, pass a list of class names to `classifier_names` to add a classification head to the CoreML model. Requires `coremltools>=9.0`, `torch>=1.11`, and `numpy<=2.3.5`. Not supported on Windows. A `BlobWriter not loaded` error at import time usually means `coremltools` has no wheel for your Python version. Use Python 3.10–3.13.

### Export to TensorFlow SavedModel

TF SavedModel export goes through ONNX as an intermediate step:

```python
from ultralytics.utils.export import torch2onnx, onnx2saved_model

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
```

The function returns a Keras model and also generates TFLite files (`.tflite`) inside `resnet18_saved_model/`. Requires `tensorflow>=2.0.0,<=2.19.0`, `onnx2tf>=1.26.3,<1.29.0`, `tf_keras<=2.19.0`, `sng4onnx>=1.0.1`, `onnx_graphsurgeon>=0.3.26` (install with `--extra-index-url https://pypi.ngc.nvidia.com`), `ai-edge-litert>=1.2.0` (`,<1.4.0` on macOS), `onnxslim>=0.1.71`, `onnx>=1.12.0,<2.0.0`, and `protobuf>=5`.

### Export to TensorFlow Frozen Graph

Building on the TF SavedModel export, you can create a frozen graph:

```python
from pathlib import Path
from ultralytics.utils.export import torch2onnx, onnx2saved_model, keras2pb

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
keras2pb(keras_model, output_file=Path("resnet18_saved_model/resnet18.pb"))
```

### Export to NCNN

```python
from ultralytics.utils.export import torch2ncnn

torch2ncnn(model, im, output_dir="resnet18_ncnn_model", device=torch.device("cpu"))
```

The directory contains fixed-name `model.ncnn.param` and `model.ncnn.bin` files along with a `model_ncnn.py` wrapper. Dependencies `ncnn` and `pnnx` are installed automatically on first use.

### Export to MNN

MNN export requires an ONNX file as input. Export to ONNX first, then convert:

```python
from ultralytics.utils.export import torch2onnx, onnx2mnn

torch2onnx(model, im, output_file="resnet18.onnx")
onnx2mnn("resnet18.onnx", output_file="resnet18.mnn")
```

Supports `half=True` for FP16 and `int8=True` for INT8 quantization. Requires `MNN>=2.9.6` and `torch>=1.10`.

### Export to PaddlePaddle

```python
from ultralytics.utils.export import torch2paddle

torch2paddle(model, im, output_dir="resnet18_paddle_model")
```

Requires `x2paddle` and the correct PaddlePaddle distribution for your platform: `paddlepaddle-gpu>=3.0.0,<3.3.0` on CUDA, `paddlepaddle==3.0.0` on ARM64 CPU, or `paddlepaddle>=3.0.0,<3.3.0` on other CPUs. Not supported on NVIDIA Jetson.

### Export to ExecuTorch

```python
from ultralytics.utils.export import torch2executorch

torch2executorch(model, im, output_dir="resnet18_executorch_model")
```

The exported `model.pte` file is saved inside `resnet18_executorch_model/`. Requires `torch>=2.9.0` and a matching ExecuTorch runtime (`pip install executorch`). For runtime usage, see the [ExecuTorch integration](../integrations/executorch.md).

## Verify Your Exported Model

After exporting, verify numerical parity with the original PyTorch model before shipping. A quick smoke test with ONNX Runtime compares outputs and flags tracing or quantization errors early:

```python
import numpy as np
import onnxruntime as ort
import torch
import timm

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pytorch_output = model(im).numpy()

session = ort.InferenceSession("resnet18.onnx")
onnx_output = session.run(None, {"images": im.numpy()})[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff:.6f}")  # should be < 1e-5
```

!!! tip "Expected difference"

    For FP32 exports, the max absolute difference should be under `1e-5`. Larger differences point to unsupported ops, incorrect input shape, or a model not in eval mode. FP16 and INT8 exports have looser tolerances. Validate on real data instead of random tensors.

For other runtimes, the input tensor name may differ. OpenVINO, for example, uses the model's forward-argument name (typically `x` for generic models), while `torch2onnx` defaults to `"images"`.

## Known Limitations

- **Multi-input support is uneven**: `torch2onnx`, `torch2openvino`, and `torch2torchscript` accept a tuple or list of example tensors for models with multiple inputs. `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` assume a single input tensor.
- **Eval mode required**: Always call `model.eval()` before exporting.
- **CoreML wheel availability**: `coremltools>=9.0` ships wheels for Python 3.10–3.13. On newer Python versions the C extension fails to load with a `BlobWriter not loaded` error. Use Python 3.10–3.13 for CoreML export.
- **ExecuTorch needs `flatc`**: The ExecuTorch runtime requires the FlatBuffers compiler. Install with `brew install flatbuffers` on macOS or `apt install flatbuffers-compiler` on Ubuntu.
- **No inference via Ultralytics**: Exported non-YOLO models cannot be loaded back through `YOLO()` for inference. Use the native runtime for each format ([ONNX Runtime](../integrations/onnx.md), [OpenVINO Runtime](../integrations/openvino.md), etc.).
- **YOLO-only formats**: [Axelera](../integrations/axelera.md) and [Sony IMX500](../integrations/sony-imx500.md) exports require YOLO-specific model attributes and are not available for generic models.
- **Platform-specific formats**: [TensorRT](../integrations/tensorrt.md) requires an NVIDIA GPU. [RKNN](../integrations/rockchip-rknn.md) requires the `rknn-toolkit2` SDK (Linux only). [Edge TPU](../integrations/edge-tpu.md) requires the `edgetpu_compiler` binary (Linux only).

## FAQ

### What models can I export with Ultralytics?

Any `torch.nn.Module`. This includes models from timm, torchvision, or any custom PyTorch model. The model must be in evaluation mode (`model.eval()`) before export. ONNX, OpenVINO, and TorchScript additionally accept a tuple of example tensors for multi-input models.

### Which export formats work without a GPU?

All supported formats (TorchScript, ONNX, OpenVINO, CoreML, TF SavedModel, NCNN, PaddlePaddle, MNN, ExecuTorch) can export on CPU. No GPU is required for the export process itself. TensorRT is the only format that requires an NVIDIA GPU.

### Why does CoreML export fail with BlobWriter error?

The error usually means `coremltools` cannot load its native C extension because no wheel is published for your Python version. `coremltools==9.0` ships wheels for Python 3.10–3.13 on macOS and Linux. Create a Python 3.10–3.13 environment to export CoreML models.

### Can I export models with multiple inputs?

Partially. `torch2onnx`, `torch2openvino`, and `torch2torchscript` accept a tuple or list of example tensors and handle multi-input models correctly. `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` still assume a single input tensor, so DETR-style models with multiple inputs will need a workaround for those formats.

### How is this different from using torch.onnx.export directly?

The Ultralytics export functions wrap `torch.onnx.export` and similar tools with sensible defaults, automatic dependency checking, and consistent logging. The main advantage is a unified API across ten formats rather than learning each tool's API separately.

### What Ultralytics version do I need?

The standalone export functions are available starting from `ultralytics>=8.4.38` following the [exporter refactor](https://github.com/ultralytics/ultralytics/pull/23914) and the [unified-args update](https://github.com/ultralytics/ultralytics/pull/24120) that standardized the `output_file` and `output_dir` parameters.

### Can I export a timm model to ONNX with Ultralytics?

Yes, timm models are standard `torch.nn.Module` instances, so they work with `torch2onnx` out of the box. Load the model with `timm.create_model(..., pretrained=True).eval()`, prepare an input tensor matching the model's expected shape, and call `torch2onnx(model, im, output_file="model.onnx")`. See the [ONNX example](#export-to-onnx) above.

### Can I export a torchvision model to CoreML for iOS deployment?

Yes. torchvision classifiers, detectors, and segmentation models export to `.mlpackage` via `torch2coreml`. For image classification models, pass a list of class names to `classifier_names` to bake in a classification head. Run the export on macOS or Linux. CoreML is not supported on Windows. See the [CoreML integration](../integrations/coreml.md) for iOS deployment details.

### Which export format should I pick for my deployment target?

It depends on your runtime environment:

| Deployment target             | Recommended format                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Cross-platform cloud / server | [ONNX](../integrations/onnx.md)                                                                                                       |
| Intel CPU or iGPU             | [OpenVINO](../integrations/openvino.md)                                                                                               |
| iOS / macOS / Apple Silicon   | [CoreML](../integrations/coreml.md)                                                                                                   |
| Android / mobile              | [TFLite](../integrations/tflite.md) (via TF SavedModel), [NCNN](../integrations/ncnn.md), [ExecuTorch](../integrations/executorch.md) |
| Browser / JavaScript          | TF.js (via TF Frozen Graph)                                                                                                           |
| Edge / embedded ARM           | [NCNN](../integrations/ncnn.md), [MNN](../integrations/mnn.md)                                                                        |
| Chinese ecosystem / Baidu     | [PaddlePaddle](../integrations/paddlepaddle.md)                                                                                       |
| PyTorch-first C++ runtime     | [TorchScript](../integrations/torchscript.md), [ExecuTorch](../integrations/executorch.md)                                            |

For NVIDIA GPUs specifically, the Ultralytics [TensorRT integration](../integrations/tensorrt.md) gives the best performance but requires YOLO-specific model attributes and is not available through these non-YOLO utilities.

### Can I quantize my exported model to INT8 or FP16?

Yes, for several formats. Pass `half=True` for FP16 or `int8=True` for INT8 when exporting to OpenVINO, CoreML, MNN, or NCNN. INT8 in OpenVINO additionally requires a `calibration_dataset` argument for [post-training quantization](https://www.ultralytics.com/glossary/model-quantization). See each format's integration page for quantization trade-offs.
