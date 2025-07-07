from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import pickle
import os
import torch

def pad_audio(audio_array, target_length=88200):  # 4 seconds at 22kHz
    """Pad or truncate audio to fixed length."""
    if len(audio_array) > target_length:
        return audio_array[:target_length]
    else:
        padding = target_length - len(audio_array)
        return torch.nn.functional.pad(torch.tensor(audio_array), (0, padding)).numpy()
    
def preprocess_dataset(dataset):
    """Preprocess audio data to fixed length."""
    print("Preprocessing audio data...")
    processed_data = []
    
    for i, sample in enumerate(dataset):
        audio_array = sample['audio']['array']
        padded_audio = pad_audio(audio_array)
        
        processed_sample = {
            'audio': torch.tensor(padded_audio, dtype=torch.float32).unsqueeze(0),  # Add channel dim
            'label': sample['classID'],
            'fold': sample['fold']
        }
        processed_data.append(processed_sample)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} samples")
    
    print(f"Preprocessing complete. Total samples: {len(processed_data)}")
    return processed_data

def load_or_cache_dataset(cache_path="dataset.pkl", streaming=True):
    """Load dataset from cache or download and cache it."""
    if os.path.exists(cache_path):
        print(f"Loading dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Downloading dataset...")
        ds = load_dataset("danavery/urbansound8K", split="train[:50]")
        preprocessed_dataset = preprocess_dataset(ds)
        print(f"Saving dataset to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(preprocessed_dataset, f)
        return preprocessed_dataset

def get_fold_splits(dataset, train_folds, val_folds, test_folds):
    """
    Split dataset by fold numbers.
    
    Args:
        dataset: The loaded dataset
        train_folds: List of fold numbers for training (e.g., [1,2,3,4,5,6,7])
        val_folds: List of fold numbers for validation (e.g., [8])
        test_folds: List of fold numbers for testing (e.g., [9,10])
    
    Returns:
        train_indices, val_indices, test_indices
    """
    # Check for overlaps
    all_folds = set(train_folds + val_folds + test_folds)
    if len(all_folds) != len(train_folds) + len(val_folds) + len(test_folds):
        raise ValueError("Fold overlap detected between train/val/test splits")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, sample in enumerate(dataset):
        fold = sample['fold']
        
        if fold in train_folds:
            train_indices.append(idx)
        elif fold in val_folds:
            val_indices.append(idx)
        elif fold in test_folds:
            test_indices.append(idx)
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices

ds = load_or_cache_dataset()