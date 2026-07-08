"""
export_onnx.py — Export YOLOX-s model to ONNX & Verify with ONNX Runtime

Migrated from RTMDet/MMDetection to YOLOX-s.
Exports the trained PyTorch weights to standard ONNX format with opset 11.
Includes automatic validation of the exported graph using onnx.checker
and execution verification using ONNX Runtime.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.yolox_surveillance import Exp


def export_and_verify(
    ckpt_path: Path,
    output_onnx_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 11,
) -> None:
    print("=" * 60)
    print("YOLOX-s ONNX Exporter & Verifier")
    print("=" * 60)
    print(f"[*] Input Checkpoint: {ckpt_path.as_posix()}")
    print(f"[*] Target ONNX Path: {output_onnx_path.as_posix()}")
    print(f"[*] Input Resolution: {input_size[0]}x{input_size[1]}")
    print(f"[*] Opset Version:    {opset_version}")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Please train the model first using: python scripts/train_model.py"
        )

    # 1. Load Experiment & Model
    print("\n[*] Step 1: Loading YOLOX-s architecture and weights...")
    exp = Exp()
    model = exp.get_model()
    
    from yolox.utils import load_ckpt
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model = load_ckpt(model, state_dict)
    
    # Enable bounding box decoding in inference so ONNX outputs directly usable boxes
    model.head.decode_in_inference = True
    model.eval()
    print("[+] Model loaded and set to evaluation mode with bbox decoding enabled.")

    # 2. Export to ONNX
    print("\n[*] Step 2: Exporting PyTorch graph to ONNX...")
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    print(f"[+] ONNX export successful: {output_onnx_path.as_posix()} ({file_size_mb:.2f} MB)")

    # 3. Verify ONNX Graph
    print("\n[*] Step 3: Verifying exported ONNX graph structure...")
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[+] onnx.checker verification passed without errors!")

    # 4. Verify ONNX Runtime Execution
    print("\n[*] Step 4: Testing ONNX Runtime inference compatibility...")
    # Prioritize GPU ExecutionProvider if available, otherwise CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(output_onnx_path), providers=providers)
        active_provider = session.get_providers()[0]
        print(f"[*] ONNX Runtime initialized with provider: {active_provider}")
    except Exception as exc:
        print(f"[!] Warning: Failed to load CUDA ExecutionProvider ({exc}). Falling back to CPU.")
        session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
        active_provider = session.get_providers()[0]
        print(f"[*] ONNX Runtime initialized with provider: {active_provider}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    test_tensor = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

    ort_inputs = {input_name: test_tensor}
    ort_outputs = session.run([output_name], ort_inputs)[0]

    print(f"[+] ONNX Runtime test execution successful!")
    print(f"    Input Shape:  {test_tensor.shape}")
    print(f"    Output Shape: {ort_outputs.shape}")
    print("=" * 60)
    print("[*] Migration verification complete. Model is ready for real-time surveillance inference!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOX-s model to ONNX")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(_REPO_ROOT / "models" / "yolox_surveillance" / "best_ckpt.pth"),
        help="Path to trained PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_REPO_ROOT / "models" / "yolox_surveillance.onnx"),
        help="Path to save exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    # If best_ckpt doesn't exist yet (not trained), fall back to pre-trained yolox_s.pth for verification
    if not ckpt_path.exists():
        fallback = _REPO_ROOT / "models" / "yolox_s.pth"
        if fallback.exists():
            print(f"[*] Target checkpoint {ckpt_path} not found. Using pre-trained {fallback} for export verification.")
            ckpt_path = fallback

    output_path = Path(args.output)
    export_and_verify(ckpt_path, output_path, opset_version=args.opset)


if __name__ == "__main__":
    main()
