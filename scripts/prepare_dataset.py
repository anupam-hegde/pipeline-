import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to point to your actual local dataset folders
FIRE_DIR = r"C:\Users\anupa\OneDrive\Documents\MEGA downloads\Merged_FireSmoke"
FIRE2_DIR = r"C:\Users\anupa\OneDrive\Documents\MEGA downloads\dataset_v2\data"
WEAPON_DIR = r"C:\Users\anupa\OneDrive\Documents\MEGA downloads\weapons\Merged"

OUTPUT_DIR = "../merged_dataset"
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation

# Class Remapping (Old ID -> New ID)
# Class 0: fire (mapped from 0/1 in fire datasets)
# Class 1: weapon (mapped from 0 in weapon dataset)
FIRE_MAP = {0: 0, 1: 0}
FIRE2_MAP = {0: 0, 1: 0}
WEAPON_MAP = {0: 1}

# Valid image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ==========================================
# DIRECTORY INITIALIZATION
# ==========================================
def create_directories():
    """Creates the YOLO directory structure inside OUTPUT_DIR."""
    out_path = Path(OUTPUT_DIR)
    
    # Subdirectories for YOLO format
    dirs_to_create = [
        out_path / "train" / "images",
        out_path / "train" / "labels",
        out_path / "val" / "images",
        out_path / "val" / "labels"
    ]
    
    # Create or overwrite structure cleanly
    if out_path.exists():
        print(f"[*] Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(out_path)
        
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        
    return out_path


# ==========================================
# PROCESSING LOGIC
# ==========================================
def process_dataset(source_dir: str, class_map: dict, prefix: str, out_path: Path):
    """
    Scans a source dataset (either direct images/labels or split train/val/test folders),
    splits into Train/Val, maps class IDs, and copies files with a prefix to prevent collision.
    """
    src_path = Path(source_dir)
    subdirs_to_check = []
    
    # Check if images/ and labels/ exist directly under source_dir
    if (src_path / "images").exists() and (src_path / "labels").exists():
        subdirs_to_check.append(src_path)
    else:
        # Check for standard split subdirectories (train, val, test, valid)
        for split in ['train', 'val', 'test', 'valid']:
            sub_img = src_path / split / "images"
            sub_lbl = src_path / split / "labels"
            if sub_img.exists() and sub_lbl.exists():
                subdirs_to_check.append(src_path / split)
                
    if not subdirs_to_check:
        print(f"[!] WARNING: Source directory {source_dir} missing 'images/' or 'labels/' subfolder(s). Skipping.")
        return 0, 0

    # Gather valid image files across all found subdirs
    valid_pairs = []
    for subdir in subdirs_to_check:
        src_images_dir = subdir / "images"
        src_labels_dir = subdir / "labels"
        for img_file in src_images_dir.iterdir():
            if img_file.suffix.lower() in VALID_EXTENSIONS:
                label_file = src_labels_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    valid_pairs.append((img_file, label_file))
                
    if not valid_pairs:
        print(f"[!] No valid image/label pairs found in {source_dir}.")
        return 0, 0

    # Random split
    random.seed(42)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * SPLIT_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    # Process split helper function
    def copy_and_remap(pairs, split_name):
        count = 0
        for img_path, lbl_path in tqdm(pairs, desc=f"Processing [{prefix}] -> {split_name}", leave=False):
            new_stem = f"{prefix}_{img_path.stem}"
            
            # 1. Copy Image
            new_img_path = out_path / split_name / "images" / (new_stem + img_path.suffix)
            shutil.copy2(img_path, new_img_path)
            
            # 2. Read, Remap, and Write Label
            new_lbl_path = out_path / split_name / "labels" / (new_stem + ".txt")
            
            remapped_lines = []
            with open(lbl_path, 'r', errors='replace') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        continue
                        
                    # Remap the ID if it exists in the dictionary, otherwise keep it or drop it
                    if old_class_id in class_map:
                        new_class_id = class_map[old_class_id]
                        # Reconstruct line: <new_class_id> <x_center> <y_center> <width> <height>
                        new_line = f"{new_class_id} " + " ".join(parts[1:])
                        remapped_lines.append(new_line)
            
            # Write new label file only if we have valid remapped annotations
            if remapped_lines:
                with open(new_lbl_path, 'w') as f:
                    f.write("\n".join(remapped_lines) + "\n")
                count += 1
            else:
                # If no valid labels were found after mapping, we remove the copied image
                # to prevent blank background images (unless intentional).
                if new_img_path.exists():
                    os.remove(new_img_path)
                
        return count

    # Execute copies
    train_count = copy_and_remap(train_pairs, "train")
    val_count = copy_and_remap(val_pairs, "val")
    
    print(f"[*] Processed [{prefix}]: {train_count} Train | {val_count} Val")
    return train_count, val_count


# ==========================================
# YAML GENERATION
# ==========================================
def generate_yaml(out_path: Path):
    """Generates the data.yaml configuration file for YOLO training."""
    yaml_path = out_path / "data.yaml"
    
    # Note: Paths in data.yaml should be relative to where YOLO is run, 
    # or absolute. Using absolute paths here to prevent issues.
    train_path = (out_path / "train").resolve().as_posix()
    val_path = (out_path / "val").resolve().as_posix()
    
    yaml_content = f"""train: {train_path}
val: {val_path}

# Classes
names:
  0: fire
  1: weapon
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"[*] Created YAML config at: {yaml_path}")


# ==========================================
# EXECUTION
# ==========================================
def main():
    print("="*50)
    print("[*] YOLO Dataset Consolidation Script")
    print("="*50)
    
    out_path = create_directories()
    
    total_train = 0
    total_val = 0
    
    # Process Fire
    print(f"\nProcessing Fire Dataset from: {FIRE_DIR}")
    t_f, v_f = process_dataset(FIRE_DIR, FIRE_MAP, prefix="fire", out_path=out_path)
    total_train += t_f
    total_val += v_f
    
    # Process Fire 2 (dataset_v2)
    print(f"\nProcessing Second Fire/Smoke Dataset from: {FIRE2_DIR}")
    t_f2, v_f2 = process_dataset(FIRE2_DIR, FIRE2_MAP, prefix="fire2", out_path=out_path)
    total_train += t_f2
    total_val += v_f2
    
    # Process Weapon
    print(f"\nProcessing Weapon Dataset from: {WEAPON_DIR}")
    t_w, v_w = process_dataset(WEAPON_DIR, WEAPON_MAP, prefix="wpn", out_path=out_path)
    total_train += t_w
    total_val += v_w
    
    # Generate YAML
    print("\nGenerating data.yaml...")
    generate_yaml(out_path)
    
    print("="*50)
    print(f"[*] Consolidation Complete!")
    print(f"Total Images -> Train: {total_train} | Validation: {total_val}")
    print(f"Merged Dataset saved to: {out_path.resolve().as_posix()}")
    print("="*50)

if __name__ == '__main__':
    main()
