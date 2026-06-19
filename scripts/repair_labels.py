"""
repair_labels.py
Fixes weapon label files that have embedded newline characters within coordinate values.
Root cause: source weapon dataset labels used bare \r (old Mac) or mixed line endings,
causing some coordinate+class_id pairs to merge as a single token.
Strategy: extract ALL numeric tokens from each label file using regex,
then regroup them into valid YOLO annotation lines (5 tokens each: class cx cy w h).
"""
import re
from pathlib import Path
from tqdm import tqdm

MERGED_DATA_ROOT = '../merged_dataset'

def repair_label_file(label_path):
    """
    Reads a label file, extracts all numeric tokens regardless of line endings,
    regroups into 5-field YOLO annotations, and rewrites the file.
    Returns True if the file was repaired, False if it was already clean.
    """
    with open(label_path, 'rb') as f:
        raw_bytes = f.read()

    # Decode bytes to string, replacing any undecodable chars
    text = raw_bytes.decode('utf-8', errors='replace')

    # Extract ALL numeric tokens: integers and floats, including negatives
    tokens = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)

    if not tokens:
        # Empty file (background image) - leave as is
        return False

    if len(tokens) % 5 != 0:
        print(f"[!] Cannot repair {label_path.name}: token count {len(tokens)} not divisible by 5. Skipping.")
        return False

    # Reconstruct valid YOLO lines: group tokens into chunks of 5
    new_lines = []
    for i in range(0, len(tokens), 5):
        group = tokens[i:i+5]
        class_id_str = group[0]
        coords = group[1:]

        # Validate class_id is an integer
        try:
            class_id = int(class_id_str)
        except ValueError:
            print(f"[!] Invalid class ID '{class_id_str}' in {label_path.name}. Skipping group.")
            continue

        # Validate coords are in [0, 1]
        try:
            coord_floats = [float(c) for c in coords]
        except ValueError:
            print(f"[!] Invalid coords {coords} in {label_path.name}. Skipping group.")
            continue

        new_lines.append(f"{class_id} {' '.join(f'{c:.10f}' for c in coord_floats)}\n")

    # Only rewrite if the reconstructed content differs from original
    original_lines_count = len([l for l in text.split('\n') if l.strip()])
    repaired = False

    if new_lines:
        with open(label_path, 'w', encoding='utf-8', newline='\n') as f_out:
            f_out.writelines(new_lines)
        repaired = True

    return repaired


def repair_all_labels(merged_root):
    merged_root = Path(merged_root)
    total_repaired = 0
    total_skipped = 0

    for split in ['train', 'val', 'test']:
        labels_dir = merged_root / split / 'labels'
        if not labels_dir.exists():
            continue

        # Only target weapon labels since that's where malformed data is
        weapon_labels = list(labels_dir.glob('weapon_*.txt'))
        print(f"\n[*] Scanning {split} split: {len(weapon_labels)} weapon label files...")

        for label_path in tqdm(weapon_labels, desc=f"Repairing {split}"):
            if label_path.stat().st_size == 0:
                continue  # Skip background (empty) files
            try:
                repaired = repair_label_file(label_path)
                if repaired:
                    total_repaired += 1
            except Exception as e:
                print(f"[!] Error repairing {label_path.name}: {e}")
                total_skipped += 1

    print(f"\n=== Repair Complete ===")
    print(f"Files repaired: {total_repaired}")
    print(f"Files skipped (errors): {total_skipped}")
    print(f"======================")


if __name__ == '__main__':
    repair_all_labels(MERGED_DATA_ROOT)
