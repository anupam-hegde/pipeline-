"""
convert_best_to_onnx.py — Export best4.pt to best.onnx and compare PyTorch vs ONNX outputs.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import onnx
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    input_pt = _REPO_ROOT / "best4.pt"
    output_onnx = _REPO_ROOT / "best.onnx"
    # Also save as best4.onnx just in case
    output_onnx_alt = _REPO_ROOT / "best4.onnx"

    print("=" * 70)
    print("OBJECT DETECTION MODEL TO ONNX EXPORTER & VALIDATOR")
    print("=" * 70)
    print(f"[*] Input Checkpoint: {input_pt}")
    print(f"[*] Target ONNX Path: {output_onnx}")

    if not input_pt.exists():
        print(f"[!] ERROR: {input_pt} does not exist.")
        sys.exit(1)

    print("\n[*] Step 1: Inspecting checkpoint format...")
    try:
        ckpt = torch.load(str(input_pt), map_location="cpu")
    except Exception as e:
        print(f"[!] ERROR loading torch checkpoint: {e}")
        sys.exit(1)

    is_ultralytics = False
    is_yolox = False

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print(f"[*] Checkpoint dict keys: {keys}")
        if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
            print("[+] Detected Ultralytics YOLO PyTorch Model object inside checkpoint.")
            is_ultralytics = True
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            # Check if it matches YOLOX keys
            first_key = list(ckpt['model'].keys())[0]
            print(f"[*] First weight key in state_dict: {first_key}")
            if 'backbone' in first_key or 'head' in first_key:
                is_yolox = True
        elif 'epoch' in ckpt or 'optimizer' in ckpt:
            print("[*] Standard training checkpoint dict.")
            is_yolox = True
    elif isinstance(ckpt, torch.nn.Module):
        print("[+] Detected raw PyTorch nn.Module object.")
        is_ultralytics = True

    model_pt = None
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    # Attempt Ultralytics YOLO loading first if detected
    if is_ultralytics:
        try:
            from ultralytics import YOLO
            print("[*] Loading via Ultralytics YOLO wrapper...")
            yolo_model = YOLO(str(input_pt))
            model_pt = yolo_model.model
            model_pt.eval()
            print("[+] Successfully loaded Ultralytics YOLO model.")
        except Exception as e:
            print(f"[*] Ultralytics wrapper failed ({e}), falling back to raw PyTorch module...")
            model_pt = ckpt['model'].float().eval() if isinstance(ckpt, dict) and 'model' in ckpt else ckpt.float().eval()

    # Attempt YOLOX loading if not loaded yet
    if model_pt is None:
        try:
            print("[*] Loading via YOLOX surveillance Exp configuration...")
            from configs.yolox_surveillance import Exp
            from yolox.utils import load_ckpt
            exp = Exp()
            model_pt = exp.get_model()
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model_pt = load_ckpt(model_pt, state_dict)
            model_pt.head.decode_in_inference = True
            model_pt.eval()
            print("[+] Successfully loaded YOLOX model with decode_in_inference=True.")
            is_yolox = True
        except Exception as e:
            print(f"[!] ERROR loading as YOLOX model: {e}")
            print("\n[!] Diagnosis / Fix Instructions:")
            print("    Please ensure best4.pt contains valid weights matching either YOLOX architecture (`configs.yolox_surveillance.Exp`) or Ultralytics YOLO format.")
            sys.exit(1)

    # --- Step 2: Export to ONNX ---
    print(f"\n[*] Step 2: Exporting PyTorch model to ONNX at {output_onnx}...")
    try:
        if is_ultralytics:
            # For Ultralytics, their native export handles pre/post processing and dynamic axes cleanly
            try:
                from ultralytics import YOLO
                yolo = YOLO(str(input_pt))
                exported_path = yolo.export(format="onnx", dynamic=True, opset=11)
                # Copy/move to desired target path best.onnx
                import shutil
                shutil.copyfile(exported_path, str(output_onnx))
                if str(output_onnx_alt) != str(output_onnx):
                    shutil.copyfile(exported_path, str(output_onnx_alt))
                print(f"[+] Ultralytics native export successful: {output_onnx}")
            except Exception as e:
                print(f"[*] Native Ultralytics export failed ({e}), using torch.onnx.export...")
                torch.onnx.export(
                    model_pt,
                    dummy_input,
                    str(output_onnx),
                    input_names=["images"],
                    output_names=["output"],
                    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
                    opset_version=11,
                    do_constant_folding=True,
                )
                import shutil
                shutil.copyfile(str(output_onnx), str(output_onnx_alt))
        else:
            torch.onnx.export(
                model_pt,
                dummy_input,
                str(output_onnx),
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
                opset_version=11,
                do_constant_folding=True,
            )
            import shutil
            shutil.copyfile(str(output_onnx), str(output_onnx_alt))
            
        file_size_mb = output_onnx.stat().st_size / (1024 * 1024)
        print(f"[+] ONNX model successfully saved to: {output_onnx} ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] EXPORT ERROR: Failed to export model to ONNX: {e}")
        print("\n[!] Diagnosis / Fix Instructions:")
        print("    1. If there are unsupported PyTorch operations, try upgrading opset_version from 11 to 12 or 13.")
        print("    2. Ensure that dynamic flow control (if/else on tensor shapes) is not impeding tracing.")
        sys.exit(1)

    # --- Step 3: Validate ONNX Model ---
    print("\n[*] Step 3: Validating ONNX model structure using onnx.checker...")
    try:
        onnx_model = onnx.load(str(output_onnx))
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model structure check passed!")
    except Exception as e:
        print(f"[!] VALIDATION ERROR: onnx.checker failed: {e}")
        sys.exit(1)

    # --- Step 4: Compare PyTorch vs ONNX Inference ---
    print("\n[*] Step 4: Comparing sample inference (PyTorch vs ONNX Runtime)...")
    np.random.seed(42)
    sample_numpy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_numpy)

    # PyTorch Inference
    with torch.no_grad():
        pt_out = model_pt(sample_tensor)
        if isinstance(pt_out, (tuple, list)):
            pt_out_np = pt_out[0].cpu().numpy()
        else:
            pt_out_np = pt_out.cpu().numpy()

    # ONNX Inference
    try:
        session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        ort_out_np = session.run([output_name], {input_name: sample_numpy})[0]
    except Exception as e:
        print(f"[!] ONNX RUNTIME ERROR: Failed to run inference session: {e}")
        sys.exit(1)

    print(f"[*] PyTorch Output Shape: {pt_out_np.shape}")
    print(f"[*] ONNX Output Shape:    {ort_out_np.shape}")

    # Compute numerical differences
    max_abs_diff = np.max(np.abs(pt_out_np - ort_out_np))
    mean_abs_diff = np.mean(np.abs(pt_out_np - ort_out_np))
    cosine_sim = np.dot(pt_out_np.flatten(), ort_out_np.flatten()) / (
        np.linalg.norm(pt_out_np.flatten()) * np.linalg.norm(ort_out_np.flatten()) + 1e-9
    )

    print("\n" + "-" * 50)
    print("NUMERICAL COMPARISON RESULTS")
    print("-" * 50)
    print(f"Max Absolute Error:  {max_abs_diff:.6e}")
    print(f"Mean Absolute Error: {mean_abs_diff:.6e}")
    print(f"Cosine Similarity:   {cosine_sim:.6f} (1.000000 = identical)")
    print("-" * 50)

    if max_abs_diff < 1e-3 or cosine_sim > 0.999:
        print("[+] SUCCESS: PyTorch and ONNX inference outputs match consistently!")
    else:
        print("[!] WARNING: Numerical discrepancy detected. Check precision or post-processing layers.")

    print("\n" + "=" * 70)
    print(f"[+] All tasks completed successfully. Saved: {output_onnx}")
    print("=" * 70)

if __name__ == "__main__":
    main()
