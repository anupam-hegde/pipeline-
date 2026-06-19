import os
from ultralytics import YOLO

def main():
    print("="*60)
    print("🚀 YOLOv11n Surveillance Pipeline - Training Script")
    print("="*60)
    
    # ---------------------------------------------------------
    # Phase 1: Model Initialization & Transfer Learning
    # ---------------------------------------------------------
    # CRITICAL REQUIREMENT: Transfer Learning
    # Instead of initializing a blank architecture (e.g., YOLO('yolo11n.yaml')), 
    # we load the pre-trained COCO weights (yolo11n.pt). 
    # The COCO dataset already contains the 'person' class. By using these weights 
    # as our base, the model retains its pre-existing high-quality understanding 
    # of human features. It only needs to learn our two new classes (fire, weapon) 
    # while fine-tuning its person detection to our specific dataset constraints.
    print("\n[*] Initializing YOLOv11n with pre-trained COCO weights...")
    model = YOLO('yolo11n.pt')

    # ---------------------------------------------------------
    # Phase 2: Fine-Tuning / Training
    # ---------------------------------------------------------
    # VRAM Safety Profile for NVIDIA RTX 3050 (6GB VRAM)
    #   - batch=8: Keeps gradient accumulation manageable for 6GB limits.
    #   - imgsz=640: Standard high-res inference size without causing OOM.
    #   - amp=True: Automatic Mixed Precision saves memory and speeds up CUDA math.
    print("\n[*] Commencing Model Training...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'merged_dataset', 'data.yaml'))
    
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,         # Force CUDA execution
        amp=True,         # Enable FP16/Mixed Precision for memory efficiency
        workers=4,        # Dataloader threads (CPU)
        project='models', # Save results to the /models directory
        name='surveillance_run'
    )
    print("\n[*] Training Phase Completed.")

    # ---------------------------------------------------------
    # Phase 3: ONNX Export for Inference Optimization
    # ---------------------------------------------------------
    print("\n[*] Reloading best weights for ONNX export...")
    # We must load the specific 'best.pt' file that the training loop just produced
    # to ensure we export the epoch with the highest validation mAP.
    best_weights_path = os.path.join('models', 'surveillance_run', 'weights', 'best.pt')
    
    if not os.path.exists(best_weights_path):
        print(f"[!] Critical Error: Could not find best weights at {best_weights_path}")
        return

    best_model = YOLO(best_weights_path)

    print("\n[*] Exporting to FP16 ONNX format...")
    # ONNX Export Parameters:
    #   - half=True: Quantizes the weights from FP32 to FP16. This halves the
    #                memory footprint on the GPU during WebSocket inference.
    #   - simplify=True: Fuses operations and optimizes the ONNX graph for faster runtime.
    best_model.export(
        format='onnx',
        half=True,
        simplify=True,
        device=0,
        imgsz=640
    )
    
    print("\n" + "="*60)
    print("✅ Pipeline Successfully Trained and Exported!")
    print(f"ONNX Model saved alongside: {best_weights_path.replace('.pt', '.onnx')}")
    print("="*60)

if __name__ == '__main__':
    main()
