from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import pickle
import os
import torch
import torchaudio

def pad_audio(audio_array, target_length=88200):  # 4 seconds at 22kHz
    """Pad or truncate audio to fixed length."""
    if len(audio_array) > target_length:
        return audio_array[:target_length]
    else:
        padding = target_length - len(audio_array)
        return torch.nn.functional.pad(torch.tensor(audio_array), (0, padding)).numpy()

def preprocess_dataset(dataset):
    """Preprocess audio data to mel spectrograms."""
    print("Preprocessing audio data to mel spectrograms...")
    processed_data = []
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0
    )
    
    for i, sample in enumerate(dataset):
        audio_array = sample['audio']['array']
        padded_audio = pad_audio(audio_array)
        
        # Convert to tensor and create mel spectrogram
        audio_tensor = torch.tensor(padded_audio, dtype=torch.float32)
        mel_spec = mel_transform(audio_tensor)
        
        # Apply log scaling for better representation
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Normalize the spectrogram
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        processed_sample = {
            'audio': log_mel_spec.unsqueeze(0),  # Add channel dim: [1, n_mels, time]
            'label': sample['classID'],
            'fold': sample['fold']
        }
        processed_data.append(processed_sample)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} samples, spec shape: {log_mel_spec.shape}")
    
    print(f"Preprocessing complete. Total samples: {len(processed_data)}")
    return processed_data

def load_or_cache_dataset(raw_cache_path="raw_dataset.pkl", processed_cache_path="processed_dataset.pkl"):
    """Load dataset from cache or download and cache it."""
    # Check if processed data exists
    if os.path.exists(processed_cache_path):
        print(f"Loading processed dataset from {processed_cache_path}")
        with open(processed_cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Check if raw data exists
    if os.path.exists(raw_cache_path):
        print(f"Loading raw dataset from {raw_cache_path}")
        with open(raw_cache_path, 'rb') as f:
            raw_ds = pickle.load(f)
    else:
        print("Downloading dataset...")
        raw_ds = load_dataset("danavery/urbansound8K", split="train")
        # raw_ds = load_dataset("danavery/urbansound8K", split="train[:20]")
        print(f"Saving raw dataset to {raw_cache_path}")
        with open(raw_cache_path, 'wb') as f:
            pickle.dump(raw_ds, f)
    
    # Process the raw data
    processed_dataset = preprocess_dataset(raw_ds)
    print(f"Saving processed dataset to {processed_cache_path}")
    with open(processed_cache_path, 'wb') as f:
        pickle.dump(processed_dataset, f)
    
    return processed_dataset

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