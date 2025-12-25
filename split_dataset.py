"""
Script to split dataset into train/val/test sets
"""

import os
import shutil
import random
import argparse


def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Directory containing all patient folders
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all patient directories
    patient_dirs = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains required files
            nii_file = os.path.join(item_path, f"{item}.nii.gz")
            pa_file = os.path.join(item_path, f"{item}_pa_drr.png")
            lat_file = os.path.join(item_path, f"{item}_lat_drr.png")
            
            if os.path.exists(nii_file) and os.path.exists(pa_file) and os.path.exists(lat_file):
                patient_dirs.append(item)
            else:
                print(f"Warning: Skipping {item} - missing required files")
    
    print(f"Found {len(patient_dirs)} complete patient datasets")
    
    # Shuffle patients
    random.seed(seed)
    random.shuffle(patient_dirs)
    
    # Calculate split indices
    n_total = len(patient_dirs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patients = patient_dirs[:n_train]
    val_patients = patient_dirs[n_train:n_train + n_val]
    test_patients = patient_dirs[n_train + n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val:   {len(val_patients)} patients")
    print(f"  Test:  {len(test_patients)} patients")
    
    # Create split directories
    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    for split_name, patients in splits.items():
        split_dir = os.path.join(source_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\nMoving {len(patients)} patients to {split_name}/...")
        
        for patient in patients:
            src = os.path.join(source_dir, patient)
            dst = os.path.join(split_dir, patient)
            
            # Move patient directory
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"  Moved: {patient}")
    
    print("\n✓ Dataset split complete!")
    print(f"\nUpdate config.yaml with:")
    print(f"  train_data_path: \"{os.path.join(source_dir, 'train')}\"")
    print(f"  val_data_path: \"{os.path.join(source_dir, 'val')}\"")
    print(f"  test_data_path: \"{os.path.join(source_dir, 'test')}\"")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing patient folders')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio for training set (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio for validation set (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio for test set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(
        args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
